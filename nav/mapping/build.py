"""Map pipeline for one place: prepare every walk -> select candidates -> LLM draft -> checked, enriched PlaceMap.

    prepare (cached per recording)   frames at 1 fps (sharpest per second), motion, voice transcripts, embeddings
    select                           quality filter, keyframes per walk, fold same-place frames across walks
                                     (MegaLoc retrieval + LightGlue check, verify.py)
    LLM                              Claude Code job (nav/mapping/llm.py) -> MapDraft
    finish                           check ids, attach evidence (recording + time + image), dead-reckoned
                                     positions aligned through shared nodes, measured walking distances, headings,
                                     status (observed / inferred), spatial relations as bearings
    publish                          reference images + their embeddings for live localization (references.py),
                                     the visual memory: every frame of every walk, where it was taken (memory.py),
                                     quality check (quality.py), human review overlay (review.py), then only what
                                     the walks back up (clean.py)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nav.mapping.continuity import previous_context, stabilize
from nav.mapping.motion import angle_diff
from nav.mapping.prepare import Prepared, prepare
from nav.mapping.select import SelectParams, cap_groups, group_frames, mark_usable, select_in_recording
from nav.mapping.verify import match_pairs
from nav.mapping.calibrate import default_fov
from nav.schemas.map import (
    MAX_INSTRUCTION_CHARS,
    BuildInfo,
    Candidate,
    CandidateSet,
    Direction,
    Evidence,
    InstructionKind,
    Link,
    MapDraft,
    MapFrame,
    Motion,
    PlaceEdge,
    PlaceMap,
    PlaceNode,
    Point,
    RecordingInfo,
    SpatialRelation,
    TimelineStep,
    Track,
    View,
    slugify,
)
from nav.schemas.recording import Manifest

log = logging.getLogger(__name__)
PIPELINE_VERSION = "mvp-2"  # mvp-2: relations, references, status, headings, calibration
USABLE_STATUS = ("complete", "incomplete")
VISIT_S = 15  # views of a node within this long of each other are one visit


@dataclass
class BuildResult:
    build_dir: Path
    candidates: CandidateSet
    map: PlaceMap | None


def recordings_for(recordings_root: Path, place: str) -> list[Path]:
    key = " ".join(place.split()).casefold()
    out = []
    for mf in sorted(recordings_root.glob("*/manifest.json")):
        m = Manifest.model_validate_json(mf.read_text())
        if m.place.casefold() == key and m.status in USABLE_STATUS and m.video:
            out.append(mf.parent)
    return out


def track_points(tr) -> list[list]:
    """A motion track at 1 Hz for tracks.json: [t_ms, x_m, y_m, heading_deg or null]."""
    idx = np.unique([tr.index(t) for t in np.arange(0, tr.t_ms[-1] + 1, 1000)])
    return [[round(float(tr.t_ms[i])), round(float(tr.x_m[i]), 2), round(float(tr.y_m[i]), 2),
             None if np.isnan(tr.heading_deg[i]) else round(float(tr.heading_deg[i]), 1)] for i in idx]


def select_candidates(place: str, preps: list[Prepared], p: SelectParams, embed_model: str, place_model: str,
                      build_dir: Path) -> CandidateSet:
    """Run selection and write candidates.json + pairs.json (same-place checks, for tuning) + frames/ into build_dir."""
    kept = []
    for prep in preps:
        mark_usable(prep.frames, p)
        kept += select_in_recording(prep.frames, prep.track, prep.marks, prep.voice, p)
    sim, same, pairs = match_pairs(kept, p)
    (build_dir / "pairs.json").write_text(json.dumps(pairs, indent=1))
    groups, hints = group_frames(kept, p, sim, same)
    groups, hints = cap_groups(groups, hints, p)
    width = max(2, len(str(len(groups))))
    ids = [f"c{i + 1:0{width}d}" for i in range(len(groups))]

    frames_dir = build_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for i, (cid, g) in enumerate(zip(ids, groups)):
        shutil.copyfile(g.rep.image_path, frames_dir / f"{cid}.jpg")
        same = [ids[b] if a == i else ids[a] for (a, b) in hints if i in (a, b)]
        candidates.append(Candidate(
            id=cid, image=f"frames/{cid}.jpg", reason=sorted({r for f in g.frames for r in f.reasons}),
            views=[View(rec_id=f.rec_id, t_ms=f.t_ms, sharpness=f.sharpness, motion=Motion(**f.motion),
                        marks=f.marks, voice=f.voice) for f in g.frames],
            possibly_same_as=sorted(set(same))))

    # Positions comparable across walks, where walks share a folded candidate (for the LLM; finish aligns again
    # through the nodes it makes).
    offsets, _ = align_recordings({c.id: c.views for c in candidates})
    for c in candidates:
        for v in c.views:
            if v.rec_id in offsets:
                v.motion.map_x_m = round(v.motion.x_m + float(offsets[v.rec_id][0]), 2)
                v.motion.map_y_m = round(v.motion.y_m + float(offsets[v.rec_id][1]), 2)

    recordings = []
    for prep in preps:
        m = prep.manifest
        views = sorted((v.t_ms, c.id, v) for c in candidates for v in c.views if v.rec_id == m.rec_id)
        timeline, prev = [], None
        for t, cid, v in views:
            if timeline and timeline[-1].candidate == cid:
                continue  # several views of one candidate in a row: one step
            heading = prep.track.mean_heading(t - 1500, t + 1500)
            before, after = prep.track.mean_heading(t - 3000, t), prep.track.mean_heading(t, t + 3000)
            turn_at = None if before is None or after is None else round(angle_diff(before, after))
            timeline.append(TimelineStep(turn_at_deg=turn_at, 
                t_ms=t, candidate=cid, walked_m=v.motion.walked_m, heading_deg=None if heading is None else round(heading),
                from_prev_m=None if prev is None else round(v.motion.walked_m - prev[0], 1),
                turn_deg=None if prev is None or prev[1] is None or heading is None else round(angle_diff(prev[1], heading))))
            prev = (v.motion.walked_m, heading)
        recordings.append(RecordingInfo(
            rec_id=m.rec_id, created_at_utc=m.created_at_utc, duration_s=m.duration_ms and round(m.duration_ms / 1000, 1),
            notes=m.notes, step_length_m=prep.track.step_length_m or m.step_length_m,
            live_step_length_m=prep.track.live_step_length_m,
            distance=prep.track.distance_source, video_distance_share=prep.track.video_share,
            motion="video" if prep.track.source == "video" else "imu",
            heading=prep.track.heading_kind, fov_deg=prep.calibration.fov_deg,
            video_offset_ms=prep.calibration.offset_ms, walked_m=round(float(prep.track.walked_m[-1]), 1),
            aligned=m.rec_id in offsets,
            frames_sampled=len(prep.frames), frames_usable=sum(f.usable for f in prep.frames),
            marks=prep.marks, voice=[{k: v[k] for k in ("t_ms", "t_end_ms", "text")} for v in prep.voice],
            timeline=timeline))
    cset = CandidateSet(place=place, map_id=slugify(place), recordings=recordings, candidates=candidates,
                        params=p.as_dict(), embed_model=embed_model,
                        embed_dim=int(kept[0].emb.shape[0]) if kept else None, place_model=place_model)
    (build_dir / "candidates.json").write_text(cset.model_dump_json(indent=1))
    tracks = {prep.manifest.rec_id: track_points(prep.track) for prep in preps}
    (build_dir / "tracks.json").write_text(json.dumps(tracks))
    return cset


# Words that only make sense to someone who can see. Spoken fields shouldn't need them (visual_cues is for that).
VISUAL_WORDS = re.compile(
    r"\b(see|sees|seen|seeing|visible|look|looks|looking|appear|appears|colou?r(?:s|ful|ed)?|red|yellow|green|blue|"
    r"white|black|grey|gray|pink|orange|purple|brown|painted|mural|bright|dim|dark|sign|signs|signage|screen|"
    r"projection|logo|letters|lit)\b", re.IGNORECASE)


def visual_word_warnings(draft: MapDraft) -> list[str]:
    """One warning per node/edge (and the summary) whose spoken text uses visual words."""
    out = []

    def check(owner: str, texts: list[str | None]) -> None:
        words = sorted({w.lower() for t in texts if t for w in VISUAL_WORDS.findall(t)})
        if words:
            out.append(f"{owner} uses visual words in spoken text: {', '.join(words)}")

    check("summary", [draft.summary])
    for n in draft.nodes:
        check(f"node {n.id}", [n.description, *n.landmarks])
    for e in draft.edges:
        check(f"edge {e.id}", [e.instruction, e.reverse_instruction, *e.landmarks])
    return out


def finish_map(draft: MapDraft, cset: CandidateSet, build: BuildInfo, build_rel: str,
               tracks: dict[str, list] | None = None) -> PlaceMap:
    """Check the draft against the candidates and add what code knows: evidence, positions, measured distances,
    and (given the walks' dead-reckoned tracks) the walked path of each edge."""
    warnings: list[str] = visual_word_warnings(draft)
    cands = {c.id: c for c in cset.candidates}

    def known(owner: str, ids: list[str]) -> list[str]:
        unknown = [c for c in ids if c not in cands]
        if unknown:
            warnings.append(f"{owner} cites unknown candidates {', '.join(unknown)}")
        return [c for c in ids if c in cands]

    nodes: list[PlaceNode] = []
    for n in draft.nodes:
        if any(x.id == n.id for x in nodes):
            warnings.append(f"duplicate node id {n.id} dropped")
            continue
        cids = known(f"node {n.id}", n.candidates)
        evidence = [Evidence(candidate=c, rec_id=v.rec_id, t_ms=v.t_ms, image=f"{build_rel}/{cands[c].image}")
                    for c in cids for v in cands[c].views]
        nodes.append(PlaceNode(**{**n.model_dump(exclude={"relations"}), "candidates": cids}, evidence=evidence))
    node_ids = {n.id for n in nodes}
    if unused := [c for c in cands if not any(c in n.candidates for n in nodes)]:
        warnings.append(f"candidates not used by any node: {', '.join(unused)}")

    edges: list[PlaceEdge] = []
    for e in draft.edges:
        if e.source not in node_ids or e.target not in node_ids or e.source == e.target:
            warnings.append(f"edge {e.id} ({e.source} -> {e.target}) dropped: unknown or identical nodes")
            continue
        if any(x.id == e.id for x in edges):
            warnings.append(f"duplicate edge id {e.id} dropped")
            continue
        edges.append(PlaceEdge(**{**e.model_dump(), "candidates": known(f"edge {e.id}", e.candidates)}))

    video_recs = {r.rec_id for r in cset.recordings if r.motion == "video"}
    offsets = _place_nodes(nodes, cands)
    warp = _loop_closure(nodes, cands, offsets, warnings)
    by_id = {n.id: n for n in nodes}
    tracks = tracks or {}
    timelines = _node_timelines(nodes, cands)
    passes = {e.id: _passes(e, timelines) for e in edges}
    for e in edges:
        _edge_walks(e, by_id[e.source], by_id[e.target], passes[e.id], tracks, warp, video_recs)
    for e in edges:
        if e.basis == "inferred" and e.walked:  # the LLM knows better that this wasn't walked
            ends = [by_id[e.source].position, by_id[e.target].position]
            e.walked, e.measured_distance_m, e.path = False, None, [p for p in ends if p is not None]
        elif e.basis == "walked" and not e.walked:
            warnings.append(f"edge {e.id} says walked, but the walk passed other spots between {e.source} and "
                            f"{e.target}: treated as inferred")
    for e in [e for e in edges if not e.walked and e.confidence == "low"]:  # a guess the app would speak as fact
        warnings.append(f"edge {e.id} ({by_id[e.source].name} -> {by_id[e.target].name}) dropped: inferred with low "
                        f"confidence")
        edges.remove(e)
    _edge_turns(edges, passes, tracks)
    for e in edges:
        steps_m = [st.distance_m for st in e.steps if st.distance_m is not None]
        e.distance_m = e.measured_distance_m if e.measured_distance_m is not None else (
            round(sum(steps_m), 1) if steps_m else None)
        if e.bidirectional and e.steps:
            src = by_id[e.source]
            e.reverse_steps = reverse_steps(e.steps, arrival=src.landmarks[0] if src.landmarks else None)
        if problem := first_step_problem(e):
            warnings.append(f"edge {e.id}: {problem}")
            e.review_flags.append(problem)
        _edge_status(e, passes[e.id], tracks)
    for n in nodes:
        n.review_flags = ([f"{n.confidence} confidence"] if n.confidence != "high" else []) + \
                         ([] if n.landmarks else ["no arrival cue"])
    fovs = {r.rec_id: r.fov_deg for r in cset.recordings if r.fov_deg}
    drafted = {n.id: n for n in draft.nodes}
    for n in nodes:
        n.relations = _relations(n, drafted[n.id].relations, cands, tracks, fovs, build_rel, warnings)
    placed = [Track(rec_id=r, points=[(p[0], *warp[r](*p[:3])) for p in tracks[r]]) for r in warp if r in tracks]
    return PlaceMap(tracks=placed, **draft.model_dump(exclude={"nodes", "edges"}), map_id=cset.map_id,
                    recordings=[r.rec_id for r in cset.recordings], build=build, nodes=nodes, edges=edges,
                    warnings=warnings, frame=map_frame(cset.recordings))


def map_frame(recordings: list[RecordingInfo]) -> MapFrame | None:
    if not recordings:
        return None
    kinds = {r.heading for r in recordings}
    source = "compass" if kinds == {"compass"} else "unknown" if None in kinds else "relative"
    live = [r.live_step_length_m for r in recordings if r.live_step_length_m]
    return MapFrame(heading_source=source, heading_zero_abs_deg=0.0 if source == "compass" else None,
                    step_length_m=round(float(np.mean([r.step_length_m for r in recordings])), 2),
                    live_step_length_m=round(float(np.mean(live)), 3) if live else None)


def _direction(rel_deg: float) -> Direction:
    b = rel_deg % 360
    return "ahead" if b < 45 or b >= 315 else "right" if b < 135 else "behind" if b < 225 else "left"


_SIDE_BEARING = {"left": -90.0, "right": 90.0, "behind": 180.0}


def _heading_before(pts: list, t_ms: float, span_ms: float = 3000) -> float | None:
    """Mean walking heading over the `span_ms` before `t_ms` (tracks.json points with headings)."""
    return _circular_mean([p[3] for p in pts if t_ms - span_ms <= p[0] < t_ms and len(p) > 3 and p[3] is not None])


def _relations(n: PlaceNode, drafted: list, cands: dict[str, Candidate], tracks: dict[str, list],
               fovs: dict[str, float], build_rel: str, warnings: list[str]) -> list[SpatialRelation]:
    """The LLM's relations as bearings: across-the-photo position x field of view, plus the camera heading of that
    photo, so the app can say left / right from the user's own heading."""
    out: dict[str, SpatialRelation] = {}
    for r in drafted:
        if r.candidate not in n.candidates:
            warnings.append(f"node {n.id}: relation to {r.subject!r} cites {r.candidate}, not one of its candidates: "
                            f"dropped")
            continue
        view = cands[r.candidate].views[0]
        fov = fovs.get(view.rec_id) or default_fov("handheld_portrait")
        if r.side == "in_photo":
            bearing = (min(max(r.x, -0.2), 1.2) - 0.5) * fov if r.x is not None else 0.0
        else:
            bearing = _SIDE_BEARING[r.side]
        vh = view.motion.heading_deg
        heading = None if vh is None else round((vh + bearing) % 360, 1)
        visit = next((v for v in _visits([w for w in _node_views(n, cands) if w.rec_id == view.rec_id])
                      if any(w.t_ms == view.t_ms for w in v)), [view])
        arrival = _heading_before(tracks.get(view.rec_id, []), min(w.t_ms for w in visit))
        ref = arrival if arrival is not None else vh
        rel = bearing if heading is None or ref is None else angle_diff(ref, heading)
        rid = f"{n.id}:{slugify(r.subject)}"
        out.setdefault(rid, SpatialRelation(
            id=rid, subject=r.subject, candidate=r.candidate, rec_id=view.rec_id, t_ms=view.t_ms,
            viewpoint_image=f"{build_rel}/{cands[r.candidate].image}", viewpoint_heading_deg=vh,
            bearing_deg=round(bearing, 1), heading_deg=heading,
            arrival_heading_deg=None if arrival is None else round(arrival % 360, 1), direction=_direction(rel),
            status=r.evidence))
    return list(out.values())


def instruction_kind(e: PlaceEdge) -> InstructionKind | None:
    """Coarse turn when leaving the source: measured if known (within 30 degrees is straight), else the first step."""
    if e.turn_deg is not None:
        t = abs(e.turn_deg)
        return "straight" if t <= 30 else "turn_around" if t >= 150 else "turn_right" if e.turn_deg > 0 else "turn_left"
    if not e.steps:
        return None
    a = e.steps[0].action
    return {"turn_left": "turn_left", "bear_left": "turn_left", "turn_right": "turn_right",
            "bear_right": "turn_right", "turn_around": "turn_around"}.get(a, "straight")


def _edge_status(e: PlaceEdge, passes: list[Pass], tracks: dict[str, list]) -> None:
    """Each direction is observed once a walk went that way, else inferred; headings when leaving the source and
    reaching the target, from the phone's heading on the best pass (or the walked path's shape without one)."""
    e.instruction_kind = instruction_kind(e)
    if not e.walked:
        e.status, e.reverse_status = "inferred", "inferred" if e.bidirectional else None
        e.review_flags.append("inferred: no walk went directly between these two")
        return
    e.status = "observed" if e.walked_forward else "inferred"
    e.reverse_status = ("observed" if e.walked_back else "inferred") if e.bidirectional else None
    if not e.walked_forward:
        e.review_flags.append("only walked from the target to the source")
    elif e.bidirectional and not e.walked_back:
        e.review_flags.append("the way back was not walked")
    best = _best_pass(passes)
    pts = tracks.get(best.rec_id, []) if best else []
    dep = arr = None
    if best and pts and len(pts[0]) >= 4:
        leave = _circular_mean([p[3] for p in pts if best.t_leave < p[0] <= best.t_leave + TURN_WINDOW_MS
                                and p[3] is not None])
        reach = _heading_before(pts, best.t_reach)
        if best.forward:
            dep, arr = leave, reach
        else:  # walked target -> source: this direction is that walk turned around
            dep = None if reach is None else reach + 180
            arr = None if leave is None else leave + 180
    elif len(e.path) >= 2:
        dep, arr = _end_heading(e.path, False), _end_heading(e.path, True)
    e.departure_heading_deg = None if dep is None else round(dep % 360, 1)
    e.arrival_heading_deg = None if arr is None else round(arr % 360, 1)


def _node_views(node: PlaceNode, cands: dict[str, Candidate]) -> list[View]:
    return [v for c in node.candidates for v in cands[c].views]


def _visits(views: list[View]) -> list[list[View]]:
    """Views of one node in one recording, split into visits (passes) by time."""
    views = sorted(views, key=lambda v: v.t_ms)
    out = [[views[0]]] if views else []
    for v in views[1:]:
        (out[-1].append(v) if v.t_ms - out[-1][-1].t_ms <= VISIT_S * 1000 else out.append([v]))
    return out


def _mean_xy(views: list[View]) -> np.ndarray:
    return np.mean([(v.motion.x_m, v.motion.y_m) for v in views], axis=0)


def align_recordings(groups: dict[str, list[View]]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Shift each recording's dead-reckoned frame onto a common one through what several recordings show (nodes, or
    candidates folded across walks): headings are compass-anchored, so only translation is fitted. The recording
    with the most groups is the reference; recordings that share nothing with the aligned ones are left out.
    Returns (shift per recording, {recording: {group: mean position of its first visit there}})."""
    local: dict[str, dict[str, np.ndarray]] = {}
    for gid, views in groups.items():
        per_rec: dict[str, list[View]] = {}
        for v in views:
            per_rec.setdefault(v.rec_id, []).append(v)
        for rec, vs in per_rec.items():
            local.setdefault(rec, {})[gid] = _mean_xy(_visits(vs)[0])
    if not local:
        return {}, local
    order = sorted(local, key=lambda r: -len(local[r]))
    offsets = {order[0]: np.zeros(2)}
    progress = True
    while progress:
        progress = False
        placed = {}
        for rec, off in offsets.items():
            for gid, xy in local[rec].items():
                placed.setdefault(gid, []).append(xy + off)
        for rec in order:
            shared = [gid for gid in local[rec] if gid in placed] if rec not in offsets else []
            if shared:
                offsets[rec] = np.mean([np.mean(placed[gid], axis=0) - local[rec][gid] for gid in shared], axis=0)
                progress = True
                break
    return offsets, local


def _place_nodes(nodes: list[PlaceNode], cands: dict[str, Candidate]) -> dict[str, np.ndarray]:
    """Dead-reckoned position per node: its first visit in each recording, recordings shifted onto each other
    through the nodes they share. Returns the per-recording shift."""
    offsets, local = align_recordings({n.id: _node_views(n, cands) for n in nodes})
    pos: dict[str, list] = {}
    for rec, off in offsets.items():
        for nid, xy in local[rec].items():
            pos.setdefault(nid, []).append(xy + off)
    for n in nodes:
        if n.id in pos:
            x, y = np.mean(pos[n.id], axis=0)
            n.position = Point(x_m=round(float(x), 1), y_m=round(float(y), 1))
    return offsets


LOOP_WARN_M = 15  # a revisit this far from the node's position by dead reckoning is worth a look


def _loop_closure(nodes: list[PlaceNode], cands: dict[str, Candidate], offsets: dict[str, np.ndarray],
                  warnings: list[str]):
    """Per recording, a function (t_ms, x, y) -> map-frame (x, y): the dead-reckoned walk, shifted into the map
    frame and bent so that every visit of a node lands on that node's position. The correction is interpolated
    in time between visits, which spreads the drift over the stretch walked in between (loop closure)."""
    warp = {}
    for rec, off in offsets.items():
        anchors = []
        for n in nodes:
            if n.position is None:
                continue
            views = [v for v in _node_views(n, cands) if v.rec_id == rec]
            for visit in _visits(views):
                t = float(np.mean([v.t_ms for v in visit]))
                d = np.array([n.position.x_m, n.position.y_m]) - (_mean_xy(visit) + off)
                anchors.append((t, d))
                if np.hypot(*d) > LOOP_WARN_M:
                    warnings.append(f"node {n.id} ({n.name}): the pass at {t / 1000:.0f} s in {rec} is "
                                    f"{np.hypot(*d):.0f} m from its position by dead reckoning (a large area, "
                                    f"drift, or two places merged)")
        anchors.sort(key=lambda a: a[0])
        ts = np.array([a[0] for a in anchors]) if anchors else np.zeros(1)
        dx = np.array([a[1][0] for a in anchors]) if anchors else np.zeros(1)
        dy = np.array([a[1][1] for a in anchors]) if anchors else np.zeros(1)

        def f(t, x, y, ts=ts, dx=dx, dy=dy, off=off):
            return (round(float(x + off[0] + np.interp(t, ts, dx)), 2),
                    round(float(y + off[1] + np.interp(t, ts, dy)), 2))
        warp[rec] = f
    return warp


@dataclass
class Pass:
    """One walk going directly between an edge's two nodes: consecutive on that walk's timeline of nodes."""

    rec_id: str
    left: list[View]  # the visit of the node it left
    reached: list[View]  # the visit of the node it reached next
    forward: bool  # source -> target (else target -> source)

    @property
    def t_leave(self) -> float:
        return max(v.t_ms for v in self.left)

    @property
    def t_reach(self) -> float:
        return min(v.t_ms for v in self.reached)

    def walked_m(self) -> float:
        """Between the centres of the two visits."""
        return abs(float(np.mean([v.motion.walked_m for v in self.reached]))
                   - float(np.mean([v.motion.walked_m for v in self.left])))


def _node_timelines(nodes: list[PlaceNode], cands: dict[str, Candidate]) -> dict[str, list[tuple[str, list[View]]]]:
    """Per recording, the visits of the map's nodes in time order: [(node id, the visit's views)]."""
    out: dict[str, list[tuple[str, list[View]]]] = {}
    for n in nodes:
        per_rec: dict[str, list[View]] = {}
        for v in _node_views(n, cands):
            per_rec.setdefault(v.rec_id, []).append(v)
        for rec, views in per_rec.items():
            out.setdefault(rec, []).extend((n.id, visit) for visit in _visits(views))
    for tl in out.values():
        tl.sort(key=lambda item: item[1][0].t_ms)
    return out


def _passes(e: PlaceEdge, timelines: dict[str, list[tuple[str, list[View]]]]) -> list[Pass]:
    """Every time a walk went directly between the edge's nodes, in either direction, over all the walks."""
    out = []
    for rec, tl in timelines.items():
        for (a, va), (b, vb) in zip(tl, tl[1:]):
            if (a, b) == (e.source, e.target):
                out.append(Pass(rec, va, vb, True))
            elif (a, b) == (e.target, e.source):
                out.append(Pass(rec, va, vb, False))
    return out


def _best_pass(passes: list[Pass], video_recs: set[str] = frozenset()) -> Pass | None:
    """The pass that stands for the edge: forward if any, then the shortest (detours and dwells are longer)."""
    def key(p: Pass):
        return (not p.forward, p.rec_id in video_recs, p.walked_m(), p.t_reach - p.t_leave)
    return min(passes, key=key) if passes else None


def _edge_walks(e: PlaceEdge, a: PlaceNode, b: PlaceNode, passes: list[Pass], tracks: dict[str, list], warp: dict,
                video_recs: set[str]) -> None:
    """How often each direction was walked, the measured distance (the shortest pass with step counts), and the
    walked path of the best pass, loop-closed. No pass at all: the LLM inferred this connection."""
    e.walked_forward = sum(p.forward for p in passes)
    e.walked_back = len(passes) - e.walked_forward
    ends = [p for p in (a.position, b.position) if p is not None]
    if not passes:
        e.walked, e.path, e.measured_distance_m = False, ends, None
        return
    e.walked = True
    measured = [p.walked_m() for p in passes if p.rec_id not in video_recs]
    e.measured_distance_m = round(min(measured), 1) if measured else None
    best = _best_pass(passes, video_recs)
    e.walked_in = best.rec_id
    e.walked_at_ms = best.t_leave if best.forward else best.t_reach  # when that walk was at the source
    if best.rec_id not in tracks or best.rec_id not in warp or len(ends) < 2:
        return
    f = warp[best.rec_id]
    inner = [Point(x_m=px, y_m=py) for p in tracks[best.rec_id] if best.t_leave < p[0] < best.t_reach
             for px, py in [f(*p[:3])]]
    if not best.forward:
        inner.reverse()  # the walk went target -> source; keep the path oriented source -> target
    e.path = [a.position, *inner, b.position]


TURN_CHECK_DEG = 45  # measured turns at least this clear are checked against the first step
TURN_WINDOW_MS = 6000  # walking direction after leaving a spot: mean heading over this long
_SIDE = {"bear_left": "left", "turn_left": "left", "bear_right": "right", "turn_right": "right"}
_INVERSE = {"bear_left": "bear_right", "bear_right": "bear_left", "turn_left": "turn_right", "turn_right": "turn_left",
            "step_up": "step_down", "step_down": "step_up", "stairs_up": "stairs_down", "stairs_down": "stairs_up"}


def first_step_problem(e: PlaceEdge) -> str | None:
    """Does the first step turn the way the walk turned when leaving the source? (walked edges, clear turns)"""
    if not (e.walked and e.steps and e.turn_deg is not None and abs(e.turn_deg) >= TURN_CHECK_DEG):
        return None
    measured = "right" if e.turn_deg > 0 else "left"
    action = e.steps[0].action
    if action == "turn_around":
        return None if abs(e.turn_deg) >= 135 else f"the first step is turn_around, but the walk turned {measured} " \
                                                    f"{abs(e.turn_deg):.0f}° when leaving {e.source}"
    if _SIDE.get(action) == measured:
        return None
    return f"the first step is {action}, but the walk turned {measured} {abs(e.turn_deg):.0f}° when leaving {e.source}"


def _swap_sides(text: str | None) -> str | None:
    if not text:
        return text
    return re.sub(r"\b(left|right|Left|Right)\b",
                  lambda m: {"left": "right", "right": "left", "Left": "Right", "Right": "Left"}[m.group(1)], text)


def reverse_steps(steps: list, arrival: str | None = None) -> list:
    """The route walked back: turn around, then each step in reverse order, undoing the turn that started the
    step after it, with sides swapped; each step ends at the cue that ended the step before it going forward."""
    from nav.schemas.map import Step

    out = []
    n = len(steps)
    for k in range(n - 1, -1, -1):
        action = "turn_around" if k == n - 1 else _INVERSE.get(steps[k + 1].action, steps[k + 1].action)
        until = steps[k - 1].until if k > 0 else arrival
        out.append(Step(action=action, distance_m=steps[k].distance_m, along=_swap_sides(steps[k].along),
                        until=until))
    return out


def _bearing(p: Point, q: Point) -> float:
    return float(np.degrees(np.arctan2(q.x_m - p.x_m, q.y_m - p.y_m)))


def _end_heading(path: list[Point], at_end: bool, span_m: float = 4.0) -> float | None:
    """Direction over the first (or last) few metres of a path."""
    seq = path[::-1] if at_end else path
    for q in seq[1:]:
        if np.hypot(q.x_m - seq[0].x_m, q.y_m - seq[0].y_m) >= span_m or q is seq[-1]:
            return _bearing(q, seq[0]) if at_end else _bearing(seq[0], q)
    return None


def _circular_mean(degs: list[float]) -> float | None:
    return float(np.degrees(np.angle(np.exp(1j * np.radians(degs)).mean()))) if degs else None


def _edge_turns(edges: list[PlaceEdge], passes: dict[str, list[Pass]], tracks: dict[str, list]) -> None:
    """Turn when leaving each walked edge's source, from the phone's heading: the heading at the end of the source
    pass (how the previous instruction leaves you standing there) vs the heading over the next seconds of walking
    toward the target. Measuring around the node's centre instead books the turn you make while arriving at a spot
    onto leaving it. Builds without headings in tracks.json fall back to the walked paths' geometry."""
    for e in edges:
        best = _best_pass([p for p in passes.get(e.id, []) if p.forward])
        if not e.walked or best is None:  # leaving the source in this direction wasn't observed
            continue
        pts = tracks.get(best.rec_id, [])
        if pts and len(pts[0]) >= 4:
            t_end = best.t_leave
            at_end = [p for p in pts if p[0] <= t_end and p[3] is not None]
            h_end = at_end[-1][3] if at_end else None  # how you stand at the end of the pass
            # not capped at the target: on short edges the turn is made while already approaching it
            h_out = _circular_mean([p[3] for p in pts if t_end < p[0] <= t_end + TURN_WINDOW_MS and p[3] is not None])
            if h_end is not None and h_out is not None:
                e.turn_deg = round(angle_diff(h_end, h_out))
    if all(len(p) >= 4 for pts in tracks.values() for p in pts[:1]) and tracks:
        return
    order = sorted((e for e in edges if e.walked and e.walked_at_ms is not None and len(e.path) >= 2),
                   key=lambda e: (e.walked_in, e.walked_at_ms))
    for prev, e in zip(order, order[1:]):
        if prev.target == e.source and prev.walked_in == e.walked_in:
            h0, h1 = _end_heading(prev.path, True), _end_heading(e.path, False)
            if h0 is not None and h1 is not None:
                e.turn_deg = round(angle_diff(h0, h1))


def build_map(place: str, *, recordings_root: Path, maps_root: Path, cache_root: Path, params: SelectParams,
              embed_model: str, place_model: str, transcribe_model: str | None, llm_model: str, language: str,
              budget_usd: float,
              run_llm: bool = True, effort: str = "medium", build_id: str | None = None,
              transcribe_language: str | None = None) -> BuildResult:
    """`build_id` (UTC "%Y%m%dT%H%M%SZ") lets a caller name the build folder before it exists, e.g. to report status.
    `transcribe_language` forces the voice notes' language; unset, each walk's is detected from its clips."""
    rec_dirs = recordings_for(recordings_root, place)
    if not rec_dirs:
        raise ValueError(f"no finished recordings with video for place {place!r}")
    preps = [prepare(d, cache_root, params, embed_model, place_model, transcribe_model, transcribe_language)
             for d in rec_dirs]
    place = preps[-1].manifest.place  # spelled as in the latest recording

    given = build_id is not None
    build_id = build_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    build_dir = maps_root / slugify(place) / "builds" / build_id
    build_dir.mkdir(parents=True, exist_ok=given)
    cset = select_candidates(place, preps, params, embed_model, place_model, build_dir)
    languages = {p.manifest.rec_id: p.voice_language for p in preps if p.voice_language}
    (build_dir / "build.json").write_text(json.dumps({"transcribe_model": transcribe_model,
                                                      "transcribe_languages": languages}))
    published = build_dir.parent.parent / "map.json"
    if published.is_file():  # what the LLM keeps stable, and what ids are matched against
        # the build's own map, not the reviewed one: hidden items keep their ids, so the review keeps hiding them
        try:
            prev = build_map_of(published.parent)[1]
        except FileNotFoundError:
            prev = PlaceMap.model_validate_json(published.read_text())
        (build_dir / "previous.json").write_text(json.dumps(previous_context(prev, cset), ensure_ascii=False, indent=1))
    log.info("%s: %d recordings -> %d candidates in %s", place, len(preps), len(cset.candidates), build_dir)
    if not run_llm:
        return BuildResult(build_dir, cset, None)
    return run_llm_on_build(build_dir, llm_model=llm_model, language=language, budget_usd=budget_usd, effort=effort)


def latest_llm_build(map_dir: Path) -> Path:
    """The newest build of a place with a saved LLM answer (what `refinish` redoes)."""
    found = sorted((map_dir / "builds").glob("*/job/claude.json"))
    if not found:
        raise FileNotFoundError(f"no build with an LLM answer in {map_dir}")
    return found[-1].parent.parent


def fork_build(build_dir: Path) -> Path:
    """A new build folder with the same candidates, frames and tracks (but no LLM answer), so rerunning the LLM
    doesn't overwrite an earlier answer."""
    new = build_dir.parent / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    shutil.copytree(build_dir, new, ignore=lambda d, names: [n for n in names if Path(d) == build_dir
                                                              and n in ("job", "map.json")])
    return new


def run_llm_on_build(build_dir: Path, *, llm_model: str, language: str, budget_usd: float,
                     effort: str = "medium") -> BuildResult:
    """LLM + finish for a build folder that has candidates.json; publishes its map as the place's map.json.
    The Claude Code job runs here, or on the inference host when NAV_INFER_URL is set (same job, same files)."""
    from nav import config
    from nav.mapping.llm import run_claude_code

    cset = CandidateSet.model_validate_json((build_dir / "candidates.json").read_text())
    if config.INFER_URL:
        from nav.mapping.inference import remote

        draft, run = remote().map_draft(build_dir, f"{cset.map_id}-{build_dir.name}", cset.place, language,
                                         llm_model, effort, budget_usd)
    else:
        draft, run = run_claude_code(build_dir, cset.place, language, llm_model, budget_usd, effort)
    return publish(build_dir, cset, draft, run)


def refinish_build(build_dir: Path, recordings_root: Path | None = None, cache_root: Path | None = None) -> BuildResult:
    """Redo the finish step from the LLM answer saved in job/claude.json (no new LLM call), e.g. after changing
    post-processing. Rebuilds tracks.json from the per-recording cache if the build predates it."""
    from nav.mapping.llm import parse_claude_result

    cset = CandidateSet.model_validate_json((build_dir / "candidates.json").read_text())
    tracks_file = build_dir / "tracks.json"
    old_tracks = json.loads(tracks_file.read_text()) if tracks_file.is_file() else {}
    if not old_tracks or any(len(pts[0]) < 4 for pts in old_tracks.values() if pts):  # missing, or no headings
        from nav.mapping.prepare import load_motion_only

        tracks_file.write_text(json.dumps({r.rec_id: track_points(load_motion_only(r.rec_id, recordings_root,
                                                                                   cache_root))
                                           for r in cset.recordings}))
    _fill_recording_info(cset, recordings_root, cache_root)
    draft, run = parse_claude_result(build_dir / "job" / "claude.json")
    old = build_dir / "map.json"
    if old.is_file():  # keep the original duration (the saved result doesn't have wall-clock time)
        prev = PlaceMap.model_validate_json(old.read_text()).build.llm
        run.duration_s, run.effort = prev.duration_s, prev.effort
    return publish(build_dir, cset, draft, run, recordings_root, cache_root)


def _fill_recording_info(cset: CandidateSet, recordings_root: Path | None, cache_root: Path | None) -> None:
    """Builds from before the navigation fields don't say which heading each walk had or its field of view: take
    them from the recording and the calibration cache (in memory; candidates.json, the build's input, stays)."""
    from nav import config
    from nav.mapping.prepare import load_walk
    from nav.mapping.select import SelectParams

    params = SelectParams(**{k: v for k, v in cset.params.items() if k in SelectParams.__dataclass_fields__})
    for r in cset.recordings:
        rec_dir = (recordings_root or config.RECORDINGS_DIR) / r.rec_id
        if (r.heading is not None and r.live_step_length_m is not None) or not (rec_dir / "manifest.json").is_file():
            continue
        try:
            walk = load_walk(rec_dir, cache_root or config.CACHE_DIR, params)
        except Exception as e:  # the map is still worth refinishing without it
            log.warning("%s: no recording info for the refinish: %s", r.rec_id, e)
            continue
        r.heading, r.fov_deg = walk.track.heading_kind, walk.calibration.fov_deg
        r.motion = "video" if walk.track.source == "video" else "imu"
        r.live_step_length_m = walk.track.live_step_length_m


def publish(build_dir: Path, cset: CandidateSet, draft: MapDraft, run, recordings_root: Path | None = None,
            cache_root: Path | None = None) -> BuildResult:
    tracks = json.loads((build_dir / "tracks.json").read_text()) if (build_dir / "tracks.json").is_file() else {}
    extra = json.loads((build_dir / "build.json").read_text()) if (build_dir / "build.json").is_file() else {}
    build_id = build_dir.name
    built_at = datetime.strptime(build_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    info = BuildInfo(build_id=build_id, built_at_utc=built_at, pipeline_version=PIPELINE_VERSION,
                     params=cset.params, embed_model=cset.embed_model, embed_dim=cset.embed_dim,
                     place_model=cset.place_model,
                     transcribe_model=extra.get("transcribe_model"),
                     transcribe_languages=extra.get("transcribe_languages") or {},
                     llm=run, candidates=len(cset.candidates))
    retired = []
    if (build_dir / "previous.json").is_file():
        draft, retired = stabilize(draft, cset, json.loads((build_dir / "previous.json").read_text()), build_id)
    pmap = finish_map(draft, cset, info, f"builds/{build_id}", tracks)
    pmap.retired = retired
    if dropped := [f"{r.id} ({r.name})" for r in retired if r.build_id == build_id and r.kind == "node"]:
        pmap.warnings.append(f"previous nodes not in this build (retired; their review entries wait): "
                             f"{', '.join(dropped)}. Check the notes: a rebuild should keep them.")
    from nav import config
    from nav.mapping.memory import add_memory
    from nav.mapping.quality import check_map
    from nav.mapping.references import add_references

    pmap = add_references(pmap, cset.params, build_dir, recordings_root or config.RECORDINGS_DIR,
                          cache_root or config.CACHE_DIR, cset.place_model or config.PLACE_MODEL)
    pmap = add_memory(pmap, cset.params, build_dir, recordings_root or config.RECORDINGS_DIR,
                      cache_root or config.CACHE_DIR, cset.place_model or config.PLACE_MODEL)
    pmap = check_map(pmap, config.QUALITY_MODEL)
    write_map(build_dir, pmap)
    return BuildResult(build_dir, cset, pmap)


def published_map(map_dir: Path, pmap: PlaceMap, clean: bool | None = None) -> PlaceMap:
    """A build's map as the place publishes it: the review applied, then cleaned (clean.py) unless turned off."""
    from nav import config
    from nav.mapping.clean import clean_map
    from nav.mapping.review import apply_review, load_review

    review = load_review(map_dir)
    out = apply_review(pmap, review)
    out.links = [Link.model_validate(e, from_attributes=True) for e in out.edges]
    for e in out.edges:  # whoever wrote it, the LLM or the review
        for field, text in (("instruction", e.instruction), ("reverse_instruction", e.reverse_instruction)):
            if text and len(text) > MAX_INSTRUCTION_CHARS:
                problem = f"{field} is {len(text)} characters: Orient speaks up to {MAX_INSTRUCTION_CHARS}, then stops"
                e.review_flags.append(problem)
                out.warnings.append(f"edge {e.id}: {problem}")
    return clean_map(out, review) if (config.MAP_CLEAN if clean is None else clean) else out


def write_map(build_dir: Path, pmap: PlaceMap) -> PlaceMap:
    """Save a build's map, and publish it with the place's review applied, cleaned, as the place's map.json."""
    (build_dir / "map.json").write_text(pmap.model_dump_json(indent=1))
    map_dir = build_dir.parent.parent
    published = published_map(map_dir, pmap)
    tmp = map_dir / ".map.json.tmp"
    tmp.write_text(published.model_dump_json(indent=1))
    tmp.replace(map_dir / "map.json")
    return published


def build_map_of(map_dir: Path) -> tuple[Path, PlaceMap]:
    """The build the place's map.json was published from, and that build's own map (before the review)."""
    build_id = PlaceMap.model_validate_json((map_dir / "map.json").read_text()).build.build_id
    build_dir = map_dir / "builds" / build_id
    return build_dir, PlaceMap.model_validate_json((build_dir / "map.json").read_text())


def remember(map_dir: Path, recordings_root: Path | None = None, cache_root: Path | None = None) -> PlaceMap:
    """(Re)build the visual memory of the published build and publish it again: no LLM call, nothing else changes
    (`nav map memory`, for maps built before memory.py)."""
    from nav import config
    from nav.mapping.memory import add_memory

    build_dir, pmap = build_map_of(map_dir)
    pmap.warnings = [w for w in pmap.warnings if not w.startswith("memory: ")]
    pmap = add_memory(pmap, pmap.build.params, build_dir, recordings_root or config.RECORDINGS_DIR,
                      cache_root or config.CACHE_DIR, pmap.build.place_model or config.PLACE_MODEL)
    return write_map(build_dir, pmap)


def republish(map_dir: Path) -> PlaceMap:
    """Apply the (changed) review and the cleaner to the published build again: no new LLM call, no recomputation."""
    build_dir, pmap = build_map_of(map_dir)
    return write_map(build_dir, pmap)
