"""Where things are and how they connect, decided by code from the walks (the LLM only names and describes them).

    walk paths   each walk's camera positions in the map frame over time: the 3D model's frames (geometry.py, 2 fps,
                 posed or placed by dead reckoning), or without a model the dead-reckoned track shifted onto the others
    spots        candidates (select.py) get the position of their views; candidates of any walk within FOLD_M of each
                 other are one spot, however different they look (the far side of a door, another walk's direction)
    nodes        the LLM's: one or more spots each, within NODE_SPAN_M of each other (a node is one place you can
                 stand at; farther candidates are left out, with a warning). Position: the mean of its spots'
    visits       a walk is at a node while its path is within the node's radius and that node is the nearest one
    edges        two nodes a walk went between directly (consecutive visits), each way it was walked; the path,
                 distance and turns of the best pass become the edge's geometry and its steps' skeleton: the turn when
                 leaving the source, then a step per bend, with distances. The LLM writes what is felt and heard along
                 them, and the spoken form
    relations    where every other node is from a node, as a bearing from the way walks arrived there
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nav.mapping.motion import angle_diff
from nav.schemas.map import Candidate, Point, SpatialRelation, Step

FOLD_M = 2.0  # candidates this close are one spot
NEARBY_M = 8.0  # other spots listed next to each candidate, for the LLM
NODE_SPAN_M = 6.0  # a node's spots are at most this far from their middle
VISIT_R_M = 2.5  # a walk is at a node within this radius (plus half the node's span)
MIN_VISIT_MS = 800  # shorter stays near a node are passing glances
JOIN_GAP_MS = 3000  # a visit interrupted this briefly is one visit
TURN_WINDOW_MS = 6000  # walking direction after leaving a node: mean heading over this long
BEND_DEG = 40  # a change of walking direction this large over BEND_SPAN_M is a bend
BEND_SPAN_M = 2.0
MIN_STEP_M = 1.0  # shorter stretches between bends are merged
REL_MAX = 6  # relations to other nodes kept per node (the nearest)


# --- walk paths ------------------------------------------------------------------------------------------------
@dataclass
class WalkPath:
    rec_id: str
    t: np.ndarray  # ms, increasing
    xy: np.ndarray  # [N, 2] map frame (nan: unknown)
    heading: np.ndarray  # [N] camera heading, deg clockwise from +y (nan: unknown)
    posed: np.ndarray  # [N] bool: from the 3D model

    def at(self, t_ms: float, max_dt_ms: float = 1500) -> tuple[float, float] | None:
        if not len(self.t):
            return None
        i = int(np.argmin(np.abs(self.t - t_ms)))
        if abs(self.t[i] - t_ms) > max_dt_ms or np.isnan(self.xy[i, 0]):
            return None
        return float(self.xy[i, 0]), float(self.xy[i, 1])

    def mean_heading(self, t0: float, t1: float) -> float | None:
        h = self.heading[(self.t >= t0) & (self.t <= t1)]
        h = h[~np.isnan(h)]
        return None if not len(h) else float(np.degrees(np.angle(np.exp(1j * np.radians(h)).mean())) % 360)


def paths_from_geometry(geo) -> dict[str, WalkPath]:
    out = {}
    recs = dict.fromkeys(f.rec_id for f in geo.frames)
    for rec in recs:
        idx = sorted((i for i, f in enumerate(geo.frames) if f.rec_id == rec), key=lambda i: geo.frames[i].t_ms)
        out[rec] = WalkPath(rec, np.array([geo.frames[i].t_ms for i in idx]), geo.xyz[idx, :2].copy(),
                            geo.heading_deg[idx].copy(), geo.posed[idx].copy())
    return out


def paths_from_tracks(tracks: dict[str, list], offsets: dict[str, np.ndarray]) -> dict[str, WalkPath]:
    """Without a model: the 1 Hz dead-reckoned tracks (tracks.json points [t, x, y, heading]), shifted onto each other
    (build.align_recordings); walks that share nothing with the others are left out."""
    out = {}
    for rec, pts in tracks.items():
        if rec not in offsets or not pts:
            continue
        a = np.array([[p[0], p[1], p[2], np.nan if len(p) < 4 or p[3] is None else p[3]] for p in pts], float)
        out[rec] = WalkPath(rec, a[:, 0], a[:, 1:3] + offsets[rec], a[:, 3], np.zeros(len(a), bool))
    return out


# --- spots -----------------------------------------------------------------------------------------------------
def candidate_positions(cands: list[Candidate], paths: dict[str, WalkPath]) -> dict[str, tuple[float, float]]:
    """Each candidate where its views were taken (the mean of the views with a position)."""
    out = {}
    for c in cands:
        got = [p for v in c.views if v.rec_id in paths and (p := paths[v.rec_id].at(v.t_ms)) is not None]
        if got:
            out[c.id] = (float(np.mean([g[0] for g in got])), float(np.mean([g[1] for g in got])))
    return out


def fold_by_position(points: list[tuple[float, float] | None]) -> list[list[int]]:
    """Groups of indexes that are one spot: complete linkage within FOLD_M (a chain of nearby frames doesn't merge a
    corridor). Items without a position stay alone."""
    groups = [[i] for i in range(len(points))]

    def far(a: list[int], b: list[int]) -> float:
        d = 0.0
        for i in a:
            for j in b:
                if points[i] is None or points[j] is None:
                    return math.inf
                d = max(d, math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]))
        return d

    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = far(groups[a], groups[b])
                if d <= FOLD_M and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            return groups
        _, a, b = best
        groups[a] += groups.pop(b)


def nearby(cid: str, pos: dict[str, tuple[float, float]]) -> list[dict]:
    """Other candidates within NEARBY_M: [{candidate, distance_m}], nearest first."""
    if cid not in pos:
        return []
    x, y = pos[cid]
    out = sorted((math.hypot(p[0] - x, p[1] - y), o) for o, p in pos.items() if o != cid)
    return [{"candidate": o, "distance_m": round(d, 1)} for d, o in out if d <= NEARBY_M]


# --- nodes -----------------------------------------------------------------------------------------------------
def node_spots(cids: list[str], pos: dict[str, tuple[float, float]]) -> tuple[list[str], list[str]]:
    """The node's candidates within NODE_SPAN_M of their middle (the candidate with the most others close), and the
    ones left out. Candidates without a position are kept."""
    placed = [c for c in cids if c in pos]
    if len(placed) <= 1:
        return cids, []
    best = max(placed, key=lambda c: sum(math.hypot(pos[c][0] - pos[o][0], pos[c][1] - pos[o][1]) <= NODE_SPAN_M
                                         for o in placed))
    keep = [c for c in cids if c not in pos or math.hypot(pos[c][0] - pos[best][0],
                                                            pos[c][1] - pos[best][1]) <= NODE_SPAN_M]
    return keep, [c for c in cids if c not in keep]


def node_position(cids: list[str], pos: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    got = [pos[c] for c in cids if c in pos]
    return None if not got else (float(np.mean([g[0] for g in got])), float(np.mean([g[1] for g in got])))


def node_radius(cids: list[str], pos: dict[str, tuple[float, float]]) -> float:
    centre = node_position(cids, pos)
    if centre is None:
        return VISIT_R_M
    spread = max((math.hypot(pos[c][0] - centre[0], pos[c][1] - centre[1]) for c in cids if c in pos), default=0.0)
    return VISIT_R_M + spread


# --- visits and passes -----------------------------------------------------------------------------------------
@dataclass
class Visit:
    node: str
    i0: int  # path indexes, inclusive
    i1: int


@dataclass
class Pass:
    """A walk going directly from node `a` to node `b`: consecutive visits."""

    rec_id: str
    a: str
    b: str
    i_leave: int  # last index at a
    i_reach: int  # first index at b
    t_leave: float
    t_reach: float

    def length_m(self, path: WalkPath) -> float:
        xy = path.xy[self.i_leave:self.i_reach + 1]
        xy = xy[~np.isnan(xy[:, 0])]
        return float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) if len(xy) > 1 else 0.0


def visits(path: WalkPath, nodes: dict[str, tuple[float, float]], radius: dict[str, float]) -> list[Visit]:
    """The walk's stays at nodes, in time order (see the module doc)."""
    ids = list(nodes)
    if not ids or not len(path.t):
        return []
    centres = np.array([nodes[n] for n in ids])
    r = np.array([radius[n] for n in ids])
    d = np.hypot(path.xy[:, None, 0] - centres[None, :, 0], path.xy[:, None, 1] - centres[None, :, 1])
    d = np.where(np.isnan(d), np.inf, d)
    near = np.argmin(d, axis=1)
    label = [ids[k] if d[i, k] <= r[k] else None for i, k in enumerate(near)]
    runs: list[Visit] = []
    for i, n in enumerate(label):
        if n is None:
            continue
        if runs and runs[-1].node == n and path.t[i] - path.t[runs[-1].i1] <= JOIN_GAP_MS:
            runs[-1].i1 = i
        else:
            runs.append(Visit(n, i, i))
    runs = [v for v in runs if path.t[v.i1] - path.t[v.i0] >= MIN_VISIT_MS or v.i0 == 0 or v.i1 == len(path.t) - 1]
    out: list[Visit] = []
    for v in runs:  # dropping glances can bring two stays at one node together
        if out and out[-1].node == v.node:
            out[-1].i1 = v.i1
        else:
            out.append(v)
    return out


def passes(paths: dict[str, WalkPath], nodes: dict[str, tuple[float, float]],
           radius: dict[str, float]) -> list[Pass]:
    out = []
    for rec, path in paths.items():
        vs = visits(path, nodes, radius)
        for v, w in zip(vs, vs[1:]):
            out.append(Pass(rec, v.node, w.node, v.i1, w.i0, float(path.t[v.i1]), float(path.t[w.i0])))
    return out


# --- edge geometry ---------------------------------------------------------------------------------------------
def turn_action(deg: float | None) -> str:
    if deg is None or abs(deg) < 20:
        return "straight"
    side = "right" if deg > 0 else "left"
    if abs(deg) < 60:
        return f"bear_{side}"
    if abs(deg) < 135:
        return f"turn_{side}"
    return "turn_around"


def _resample(xy: np.ndarray, step_m: float = 0.5) -> np.ndarray:
    xy = xy[~np.isnan(xy[:, 0])]
    if len(xy) < 2:
        return xy
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step_m:
        return xy[[0, -1]]
    grid = np.arange(0, s[-1] + 1e-9, step_m)
    return np.stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])], axis=1)


def _bearing(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])) % 360)


def skeleton(xy: np.ndarray, departure_turn: float | None) -> list[Step]:
    """The edge's steps from its walked path: the turn when leaving, then a step per bend, each with its metres."""
    pts = _resample(xy)
    if len(pts) < 2:
        return [Step(action=turn_action(departure_turn), distance_m=None, along=None, until=None)]
    k = max(1, int(BEND_SPAN_M / 0.5))
    bends: list[tuple[int, float]] = []
    i = k
    while i < len(pts) - k:
        change = angle_diff(_bearing(pts[i - k], pts[i]), _bearing(pts[i], pts[i + k]))
        if abs(change) >= BEND_DEG:
            j = max(range(i, min(i + k, len(pts) - k)),
                    key=lambda j: abs(angle_diff(_bearing(pts[j - k], pts[j]), _bearing(pts[j], pts[j + k]))))
            change = angle_diff(_bearing(pts[j - k], pts[j]), _bearing(pts[j], pts[j + k]))
            bends.append((j, change))
            i = j + k
        else:
            i += 1
    cuts = [0] + [j for j, _ in bends] + [len(pts) - 1]
    lengths = [0.5 * (b - a) for a, b in zip(cuts, cuts[1:])]
    actions = [turn_action(departure_turn)] + [turn_action(c) for _, c in bends]
    steps: list[tuple[str, float]] = []
    for action, length in zip(actions, lengths):
        if steps and length < MIN_STEP_M and action != "turn_around":
            steps[-1] = (steps[-1][0], steps[-1][1] + length)  # too short to say: part of the previous stretch
        else:
            steps.append((action, length))
    return [Step(action=a, distance_m=round(max(m, 0.5) * 2) / 2, along=None, until=None) for a, m in steps]


@dataclass
class EdgeGeometry:
    source: str
    target: str
    passes: list[Pass]  # every pass, either way
    best: Pass  # the one that stands for the edge (forward if any, then the shortest)
    path: list[Point]
    distance_m: float
    turn_deg: float | None
    departure_deg: float | None
    arrival_deg: float | None
    steps: list[Step]
    candidates: list[str] = field(default_factory=list)  # candidate frames along the best pass, in walk order

    @property
    def walked_forward(self) -> int:
        return sum(p.a == self.source for p in self.passes)

    @property
    def walked_back(self) -> int:
        return sum(p.a == self.target for p in self.passes)


def edge_geometry(source: str, target: str, ps: list[Pass], paths: dict[str, WalkPath],
                  positions: dict[str, tuple[float, float]], cands: list[Candidate]) -> EdgeGeometry:
    forward = [p for p in ps if p.a == source]
    best = min(forward or ps, key=lambda p: (p.length_m(paths[p.rec_id]), p.t_reach - p.t_leave))
    path = paths[best.rec_id]
    xy = path.xy[best.i_leave:best.i_reach + 1]
    if best.a != source:
        xy = xy[::-1]
    ends = [positions[source], *[(float(x), float(y)) for x, y in xy if not np.isnan(x)], positions[target]]
    pts = [Point(x_m=round(x, 2), y_m=round(y, 2)) for x, y in ends]
    distance = min(p.length_m(paths[p.rec_id]) for p in ps)
    turn = dep = arr = None
    if best.a == source:
        h_end = path.mean_heading(best.t_leave - 1000, best.t_leave)  # how you stand there
        h_out = path.mean_heading(best.t_leave + 1, best.t_leave + TURN_WINDOW_MS)
        if h_end is not None and h_out is not None:
            turn = round(angle_diff(h_end, h_out))
        dep, arr = h_out, path.mean_heading(best.t_reach - 3000, best.t_reach)
    else:  # only walked target -> source: this direction is that walk turned around
        leave = path.mean_heading(best.t_leave + 1, best.t_leave + TURN_WINDOW_MS)
        reach = path.mean_heading(best.t_reach - 3000, best.t_reach)
        dep = None if reach is None else (reach + 180) % 360
        arr = None if leave is None else (leave + 180) % 360
    steps = skeleton(np.array(ends), turn)
    along = sorted((v.t_ms, c.id) for c in cands for v in c.views
                   if v.rec_id == best.rec_id and best.t_leave <= v.t_ms <= best.t_reach)
    cids = list(dict.fromkeys(c for _, c in along))
    if best.a != source:
        cids.reverse()
    return EdgeGeometry(source, target, ps, best, pts, round(distance, 1), turn,
                        None if dep is None else round(dep % 360, 1), None if arr is None else round(arr % 360, 1),
                        steps, cids)


def edges_from_passes(ps: list[Pass], paths: dict[str, WalkPath], positions: dict[str, tuple[float, float]],
                      cands: list[Candidate], orient: dict[frozenset, tuple[str, str]] | None = None
                      ) -> list[EdgeGeometry]:
    """One edge per pair of nodes walked between, oriented as `orient` has the pair (the previous map), else the way
    it was first walked."""
    pairs: dict[frozenset, list[Pass]] = {}
    for p in sorted(ps, key=lambda p: (p.rec_id, p.t_leave)):
        pairs.setdefault(frozenset((p.a, p.b)), []).append(p)
    out = []
    for key, group in pairs.items():
        s, t = (orient or {}).get(key) or (group[0].a, group[0].b)
        out.append(edge_geometry(s, t, group, paths, positions, cands))
    return out


# --- relations -------------------------------------------------------------------------------------------------
def arrival_heading(node: str, ps: list[Pass], paths: dict[str, WalkPath]) -> float | None:
    """How walks faced when they arrived at the node (circular mean over the arrivals)."""
    hs = [h for p in ps if p.b == node and (h := paths[p.rec_id].mean_heading(p.t_reach - 3000, p.t_reach)) is not None]
    return None if not hs else float(np.degrees(np.angle(np.exp(1j * np.radians(hs)).mean())) % 360)


def node_relations(node_id: str, names: dict[str, str], positions: dict[str, tuple[float, float]],
                   arrival: float | None, view: tuple[str, str, float, str, float | None]) -> list[SpatialRelation]:
    """Where the nearest other nodes are from this one: their bearing in the map frame, and ahead / left / right /
    behind from the way walks arrived (from `view`'s camera heading without arrivals). `view`: the node's photo
    (candidate, rec_id, t_ms, image, camera heading)."""
    from nav.mapping.build import _direction
    from nav.schemas.map import slugify

    if node_id not in positions:
        return []
    x, y = positions[node_id]
    cand, rec, t_ms, image, vh = view
    out = []
    others = sorted((math.hypot(p[0] - x, p[1] - y), o) for o, p in positions.items() if o != node_id)
    for d, o in others[:REL_MAX]:
        heading = _bearing(np.array([x, y]), np.array(positions[o]))
        ref = arrival if arrival is not None else vh
        rel = 0.0 if ref is None else angle_diff(ref, heading)
        out.append(SpatialRelation(
            id=f"{node_id}:{slugify(names[o])}", subject=names[o], candidate=cand, rec_id=rec, t_ms=t_ms,
            viewpoint_image=image, viewpoint_heading_deg=vh,
            bearing_deg=round(0.0 if vh is None else angle_diff(vh, heading), 1), heading_deg=round(heading, 1),
            arrival_heading_deg=None if arrival is None else round(arrival, 1), direction=_direction(rel),
            status="observed"))
    return out
