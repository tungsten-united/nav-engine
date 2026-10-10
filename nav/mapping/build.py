"""Map pipeline for one place: prepare every walk -> select candidates -> LLM draft -> checked, enriched PlaceMap.

    prepare (cached per recording)   frames at 1 fps (sharpest per second), motion, voice transcripts, embeddings
    select                           quality filter, keyframes per walk, fold same-place frames across walks
                                     (MegaLoc retrieval + LightGlue check, verify.py)
    LLM                              Claude Code job (nav/mapping/llm.py) -> MapDraft
    finish                           check ids, attach evidence (recording + time + image), dead-reckoned
                                     positions aligned through shared nodes, measured walking distances
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nav.mapping.motion import angle_diff
from nav.mapping.prepare import Prepared, prepare
from nav.mapping.select import SelectParams, cap_groups, group_frames, mark_usable, select_in_recording
from nav.mapping.verify import match_pairs
from nav.schemas.map import (
    BuildInfo,
    Candidate,
    CandidateSet,
    Evidence,
    MapDraft,
    Motion,
    PlaceEdge,
    PlaceMap,
    PlaceNode,
    Point,
    RecordingInfo,
    TimelineStep,
    Track,
    View,
    slugify,
)
from nav.schemas.recording import Manifest

log = logging.getLogger(__name__)
PIPELINE_VERSION = "mvp-1"
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
            notes=m.notes, step_length_m=m.step_length_m, walked_m=round(float(prep.track.walked_m[-1]), 1),
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
        nodes.append(PlaceNode(**{**n.model_dump(), "candidates": cids}, evidence=evidence))
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

    offsets = _place_nodes(nodes, cands)
    warp = _loop_closure(nodes, cands, offsets, warnings)
    by_id = {n.id: n for n in nodes}
    tracks = tracks or {}
    used = {c for n in nodes for c in n.candidates}
    for e in edges:
        e.measured_distance_m = _measured(by_id[e.source], by_id[e.target], cands)
        _edge_path(e, by_id[e.source], by_id[e.target], cands, tracks, warp, used)
    for e in edges:
        if e.basis == "inferred" and e.walked:  # the LLM knows better that this wasn't walked
            ends = [by_id[e.source].position, by_id[e.target].position]
            e.walked, e.measured_distance_m, e.path = False, None, [p for p in ends if p is not None]
        elif e.basis == "walked" and not e.walked:
            warnings.append(f"edge {e.id} says walked, but the walk passed other spots between {e.source} and "
                            f"{e.target}: treated as inferred")
    _edge_turns(edges, by_id, cands, tracks)
    for e in edges:
        steps_m = [st.distance_m for st in e.steps if st.distance_m is not None]
        e.distance_m = e.measured_distance_m if e.measured_distance_m is not None else (
            round(sum(steps_m), 1) if steps_m else None)
        if e.bidirectional and e.steps:
            src = by_id[e.source]
            e.reverse_steps = reverse_steps(e.steps, arrival=src.landmarks[0] if src.landmarks else None)
        if problem := first_step_problem(e):
            warnings.append(f"edge {e.id}: {problem}")
    placed = [Track(rec_id=r, points=[(p[0], *warp[r](*p[:3])) for p in tracks[r]]) for r in warp if r in tracks]
    return PlaceMap(tracks=placed, **draft.model_dump(exclude={"nodes", "edges"}), map_id=cset.map_id,
                    recordings=[r.rec_id for r in cset.recordings], build=build, nodes=nodes, edges=edges,
                    warnings=warnings)


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


def _place_nodes(nodes: list[PlaceNode], cands: dict[str, Candidate]) -> dict[str, np.ndarray]:
    """Dead-reckoned position per node: its first visit in each recording, recordings shifted onto each other
    through the nodes they share (headings are compass-anchored, so only translation is fitted). Returns the
    per-recording shift."""
    local: dict[str, dict[str, np.ndarray]] = {}
    for n in nodes:
        per_rec: dict[str, list[View]] = {}
        for v in _node_views(n, cands):
            per_rec.setdefault(v.rec_id, []).append(v)
        for rec, views in per_rec.items():
            local.setdefault(rec, {})[n.id] = _mean_xy(_visits(views)[0])
    if not local:
        return {}
    order = sorted(local, key=lambda r: -len(local[r]))
    offsets = {order[0]: np.zeros(2)}
    progress = True
    while progress:
        progress = False
        placed = {}
        for rec, off in offsets.items():
            for nid, xy in local[rec].items():
                placed.setdefault(nid, []).append(xy + off)
        for rec in order:
            shared = [nid for nid in local[rec] if nid in placed] if rec not in offsets else []
            if shared:
                offsets[rec] = np.mean([np.mean(placed[nid], axis=0) - local[rec][nid] for nid in shared], axis=0)
                progress = True
                break
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


def _edge_path(e: PlaceEdge, a: PlaceNode, b: PlaceNode, cands: dict[str, Candidate], tracks: dict[str, list],
               warp: dict, used: set[str]) -> None:
    """Path of an edge: the stretch of a walk between the closest-in-time views of its two nodes, loop-closed."""
    pairs = [(abs(x.t_ms - y.t_ms), x, y) for x in _node_views(a, cands) for y in _node_views(b, cands)
             if x.rec_id == y.rec_id and x.rec_id in tracks and x.rec_id in warp]
    if not pairs or a.position is None or b.position is None:
        return
    _, va, vb = min(pairs, key=lambda p: p[0])
    lo, hi = sorted((va.t_ms, vb.t_ms))
    e.walked_at_ms, e.walked_in = va.t_ms, va.rec_id
    # Not walked directly if the walk passed another node in between: the LLM inferred this connection.
    others = [v for c in cands.values() for v in c.views if v.rec_id == va.rec_id and lo < v.t_ms < hi
              and c.id not in a.candidates and c.id not in b.candidates and c.id in used]
    if others:
        e.walked, e.path, e.measured_distance_m = False, [a.position, b.position], None
        return
    f = warp[va.rec_id]
    inner = [Point(x_m=px, y_m=py) for p in tracks[va.rec_id] if lo < p[0] < hi for px, py in [f(*p[:3])]]
    if va.t_ms > vb.t_ms:
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


def _edge_turns(edges: list[PlaceEdge], by_id: dict[str, PlaceNode], cands: dict[str, Candidate],
                tracks: dict[str, list]) -> None:
    """Turn when leaving each walked edge's source, from the phone's heading: the heading at the end of the source
    pass (how the previous instruction leaves you standing there) vs the heading over the next seconds of walking
    toward the target. Measuring around the node's centre instead books the turn you make while arriving at a spot
    onto leaving it. Builds without headings in tracks.json fall back to the walked paths' geometry."""
    for e in edges:
        if not e.walked:
            continue
        visits = _closest_visits(by_id[e.source], by_id[e.target], cands)
        pts = tracks.get(e.walked_in or "", [])
        if visits and pts and len(pts[0]) >= 4:
            src, tgt = visits
            t_end, t_next = max(v.t_ms for v in src), min(v.t_ms for v in tgt)
            if t_next <= t_end:  # walked the other way: leaving the source in this direction wasn't observed
                continue
            at_end = [p for p in pts if p[0] <= t_end and p[3] is not None]
            h_end = at_end[-1][3] if at_end else None  # how you stand at the end of the pass
            # not capped at the target: on short edges the turn is made while already approaching it
            h_out = _circular_mean([p[3] for p in pts if t_end < p[0] <= t_end + TURN_WINDOW_MS and p[3] is not None])
            if h_end is not None and h_out is not None:
                e.turn_deg = round(angle_diff(h_end, h_out))
            continue
    if all(len(p) >= 4 for pts in tracks.values() for p in pts[:1]) and tracks:
        return
    order = sorted((e for e in edges if e.walked and e.walked_at_ms is not None and len(e.path) >= 2),
                   key=lambda e: (e.walked_in, e.walked_at_ms))
    for prev, e in zip(order, order[1:]):
        if prev.target == e.source and prev.walked_in == e.walked_in:
            h0, h1 = _end_heading(prev.path, True), _end_heading(e.path, False)
            if h0 is not None and h1 is not None:
                e.turn_deg = round(angle_diff(h0, h1))


def _closest_visits(a: PlaceNode, b: PlaceNode, cands: dict[str, Candidate]) -> tuple[list[View], list[View]] | None:
    """The visits of a and b closest in time within one recording (a node can be passed several times)."""
    va_all, vb_all = _node_views(a, cands), _node_views(b, cands)
    pairs = [(abs(x.t_ms - y.t_ms), x, y) for x in va_all for y in vb_all if x.rec_id == y.rec_id]
    if not pairs:
        return None
    _, x, y = min(pairs, key=lambda p: p[0])
    visit = lambda views, v: [w for w in views if w.rec_id == v.rec_id and abs(w.t_ms - v.t_ms) <= VISIT_S * 1000]
    return visit(va_all, x), visit(vb_all, y)


def _measured(a: PlaceNode, b: PlaceNode, cands: dict[str, Candidate]) -> float | None:
    """Walked distance between the centres of the two nodes' closest visits."""
    visits = _closest_visits(a, b, cands)
    if not visits:
        return None
    wa, wb = (float(np.mean([v.motion.walked_m for v in vs])) for vs in visits)
    return round(abs(wa - wb), 1)


def build_map(place: str, *, recordings_root: Path, maps_root: Path, cache_root: Path, params: SelectParams,
              embed_model: str, place_model: str, transcribe_model: str | None, llm_model: str, language: str,
              budget_usd: float,
              run_llm: bool = True, effort: str = "medium") -> BuildResult:
    rec_dirs = recordings_for(recordings_root, place)
    if not rec_dirs:
        raise ValueError(f"no finished recordings with video for place {place!r}")
    preps = [prepare(d, cache_root, params, embed_model, place_model, transcribe_model) for d in rec_dirs]
    place = preps[-1].manifest.place  # spelled as in the latest recording

    build_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    build_dir = maps_root / slugify(place) / "builds" / build_id
    build_dir.mkdir(parents=True)
    cset = select_candidates(place, preps, params, embed_model, place_model, build_dir)
    (build_dir / "build.json").write_text(json.dumps({"transcribe_model": transcribe_model}))
    log.info("%s: %d recordings -> %d candidates in %s", place, len(preps), len(cset.candidates), build_dir)
    if not run_llm:
        return BuildResult(build_dir, cset, None)
    return run_llm_on_build(build_dir, llm_model=llm_model, language=language, budget_usd=budget_usd, effort=effort)


def fork_build(build_dir: Path) -> Path:
    """A new build folder with the same candidates, frames and tracks (but no LLM answer), so rerunning the LLM
    doesn't overwrite an earlier answer."""
    new = build_dir.parent / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    shutil.copytree(build_dir, new, ignore=lambda d, names: [n for n in names if Path(d) == build_dir
                                                              and n in ("job", "map.json")])
    return new


def run_llm_on_build(build_dir: Path, *, llm_model: str, language: str, budget_usd: float,
                     effort: str = "medium") -> BuildResult:
    """LLM + finish for a build folder that has candidates.json; publishes its map as the place's map.json."""
    from nav.mapping.llm import run_claude_code

    cset = CandidateSet.model_validate_json((build_dir / "candidates.json").read_text())
    draft, run = run_claude_code(build_dir, cset.place, language, llm_model, budget_usd, effort)
    return publish(build_dir, cset, draft, run)


def refinish_build(build_dir: Path) -> BuildResult:
    """Redo the finish step from the LLM answer saved in job/claude.json (no new LLM call), e.g. after changing
    post-processing. Rebuilds tracks.json from the per-recording cache if the build predates it."""
    from nav.mapping.llm import parse_claude_result

    cset = CandidateSet.model_validate_json((build_dir / "candidates.json").read_text())
    tracks_file = build_dir / "tracks.json"
    old_tracks = json.loads(tracks_file.read_text()) if tracks_file.is_file() else {}
    if not old_tracks or any(len(pts[0]) < 4 for pts in old_tracks.values() if pts):  # missing, or no headings
        from nav.mapping.prepare import load_motion_only

        tracks_file.write_text(json.dumps({r.rec_id: track_points(load_motion_only(r.rec_id))
                                           for r in cset.recordings}))
    draft, run = parse_claude_result(build_dir / "job" / "claude.json")
    old = build_dir / "map.json"
    if old.is_file():  # keep the original duration (the saved result doesn't have wall-clock time)
        prev = PlaceMap.model_validate_json(old.read_text()).build.llm
        run.duration_s, run.effort = prev.duration_s, prev.effort
    return publish(build_dir, cset, draft, run)


def publish(build_dir: Path, cset: CandidateSet, draft: MapDraft, run) -> BuildResult:
    tracks = json.loads((build_dir / "tracks.json").read_text()) if (build_dir / "tracks.json").is_file() else {}
    extra = json.loads((build_dir / "build.json").read_text()) if (build_dir / "build.json").is_file() else {}
    build_id = build_dir.name
    built_at = datetime.strptime(build_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    info = BuildInfo(build_id=build_id, built_at_utc=built_at, pipeline_version=PIPELINE_VERSION,
                     params=cset.params, embed_model=cset.embed_model, embed_dim=cset.embed_dim,
                     place_model=cset.place_model,
                     transcribe_model=extra.get("transcribe_model"),
                     llm=run, candidates=len(cset.candidates))
    pmap = finish_map(draft, cset, info, f"builds/{build_id}", tracks)
    from nav import config
    from nav.mapping.quality import check_map

    pmap = check_map(pmap, config.QUALITY_MODEL)
    write_map(build_dir, pmap)
    return BuildResult(build_dir, cset, pmap)


def write_map(build_dir: Path, pmap: PlaceMap) -> None:
    """Save a build's map and publish it as the place's map.json."""
    text = pmap.model_dump_json(indent=1)
    (build_dir / "map.json").write_text(text)
    map_dir = build_dir.parent.parent
    tmp = map_dir / ".map.json.tmp"
    tmp.write_text(text)
    tmp.replace(map_dir / "map.json")
