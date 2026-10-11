"""Visual place recognition against a map's reference images (docs/navigation_stack.md 4.1 and 6): live frames ->
ranked nodes and a confidence status. Stateless: agreement across consecutive results (4.2) is the caller's (Orient's
orchestrator, which follows its route on them: replay.py `OrchestratorLoop` models it).

Each live frame is a place embedding (the map's `embedding.model`, MegaLoc), compared by cosine with every reference
image of the map (`embeddings.npy`, unit rows). A burst of frames scores each reference by their mean similarity. A
node scores as its best reference; the margin is the best node's score minus the second's.

With a visual memory (PlaceMap.memory, nav/mapping/memory.py: every frame of every walk and where it was taken), the
frames are also compared with every memory frame. The best one gives `position`; a memory frame counts for the node
whose region it is in (the nearest node to where it was taken, within `MEMORY_RADIUS_M`), so a node scores as its
best reference or memory frame. Regions come from positions, not from what the frame shows, so one view can't
belong to two nodes in two walks (Itnig's area nodes did, which kept every lead small).

Verification (`verify`, nav-api: LightGlue on the inference host): the best few memory frames (`VerifyParams.k`, a
few seconds apart in their walks) are matched with the camera frames, and the inliers that fit one geometry add up to
`gain` to that frame's score (from `low` to `high` inliers; live frames are matched at the memory's 768 px). A cosine lead of 0.03 is a guess; 300 inliers on the
same structure is not (Itnig, cross-walk: within 3 m of the matched frame 80% of the time). The belief downstream
reads score differences, so a verified frame moves it in one or two calls instead of never. Without an answer in
time (`verify` returns None), the result is the unverified one.

Status, preferring a missed confirmation to a false one:
- lost: the best score is under `lost_score` (nothing recognised);
- confirmed: best score >= `confirm_score`, margin >= `confirm_margin`, and, when the last confirmed node is known,
  the best node is that node or one edge away from it (graph consistency);
- uncertain: anything else.

With the phone's motion (`motions`, one per frame, the Orient app's `Motion`; all optional):
- frames pointed at the floor or ceiling (|camera elevation| > max_abs_pitch_deg, from the orientation angles) are
  left out of the mean, as the map's references are; with none left the result is lost;
- walked-distance gate: with `previous` and `previous_step_count` (the phone's stepCount when the user's node became
  `previous`; re-confirming the same node doesn't reset it), walked = new steps x the map's live step length
  (metres per phone step). A node d metres away along the map's edges is plausible once
  walked >= d - max(gate_slack_m, gate_slack_frac d); a best node not plausible stays uncertain. That is what keeps
  a place recognised from afar from moving the user there. It replaces the "one edge from previous" rule: a node
  two edges away is fine once walked that far, so a missed checkpoint no longer stalls the rest of the route. A step
  count below `previous_step_count` (the phone restarted its count) leaves the gate out: the one-edge rule applies;
- heading: the newest motion's compass replaces `heading_abs_deg` (reported against each reference, not scored).
The default thresholds are placeholders until tuned on venue images (env NAV_LOC_*; nav/config.py).
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np

from nav.schemas.map import MemoryFrame, PlaceMap
from nav.schemas.navigation import LiveMotion, LivePosition, Localization, NodeMatch, RefMatch, Thresholds

HEADING_MAX_ACCURACY_DEG = 45.0  # a compass reporting a worse accuracy is not used
MEMORY_RADIUS_M = 5.0  # a memory frame belongs to the nearest node within this distance (else to none)
MEMORY_VISIT_MS = 3000  # a frame of a walk that isn't placed: the node that walk was at within this long


@dataclass
class MemoryIndex:
    """A map's visual memory, ready to match: row i of `vectors` is `frames[i]` (positions in the published map's
    frame, i.e. scaled as the review scaled the map), `nodes[i]` the node whose region it is in."""

    frames: list[MemoryFrame]
    vectors: np.ndarray  # [M, D] float32 unit rows
    nodes: list[str | None]

    @classmethod
    def build(cls, pmap: PlaceMap, frames: list[MemoryFrame], vectors: np.ndarray, radius_m: float = MEMORY_RADIUS_M,
              exclude_rec: str | None = None) -> "MemoryIndex":
        k = pmap.frame.distance_scale if pmap.frame else 1.0
        placed = [(n.id, n.position.x_m, n.position.y_m) for n in pmap.nodes if n.position is not None]
        xy = np.array([(x, y) for _, x, y in placed], float).reshape(-1, 2)
        visits: dict[str, list[tuple[float, str]]] = {}  # rec -> [(t_ms, node)]: for walks without a position
        for n in pmap.nodes:
            for ev in n.evidence:
                visits.setdefault(ev.rec_id, []).append((ev.t_ms, n.id))
        keep, nodes = [], []
        for i, f in enumerate(frames):
            if f.rec_id == exclude_rec:
                continue
            f = f.model_copy(update={"x_m": None if f.x_m is None else round(f.x_m * k, 2),
                                     "y_m": None if f.y_m is None else round(f.y_m * k, 2)})
            node = None
            if f.x_m is not None and len(xy):
                d = np.hypot(xy[:, 0] - f.x_m, xy[:, 1] - f.y_m)
                j = int(np.argmin(d))
                node = placed[j][0] if d[j] <= radius_m else None
            elif f.x_m is None:
                near = min(visits.get(f.rec_id, []), key=lambda v: abs(v[0] - f.t_ms), default=None)
                node = near[1] if near and abs(near[0] - f.t_ms) <= MEMORY_VISIT_MS else None
            keep.append((f, i))
            nodes.append(node)
        rows = [i for _, i in keep]
        dim = vectors.shape[1] if vectors.ndim == 2 else 0
        return cls([f for f, _ in keep], vectors[rows].astype(np.float32) if rows else np.zeros((0, dim), np.float32),
                   nodes)


@dataclass
class VerifyParams:
    k: int = 6  # memory frames verified per call
    low: int = 150  # inliers below this add nothing ...
    high: int = 500  # ... and from this on the whole gain
    gain: float = 0.3
    spread_ms: float = 2000  # two verified frames of one walk are at least this far apart

    def confidence(self, inliers: int) -> float:
        return min(max((inliers - self.low) / max(self.high - self.low, 1), 0.0), 1.0)


def _diverse_top(scores: np.ndarray, frames: list[MemoryFrame], k: int, spread_ms: float) -> list[int]:
    """The k best rows, skipping rows within spread_ms of a chosen one in the same walk (the same view again)."""
    out: list[int] = []
    for i in np.argsort(-scores):
        f = frames[i]
        if all(frames[j].rec_id != f.rec_id or abs(frames[j].t_ms - f.t_ms) >= spread_ms for j in out):
            out.append(int(i))
            if len(out) >= k:
                break
    return out


@dataclass
class _MemoryRef:
    """A memory frame where NodeMatch lists references."""

    id: str
    image: str
    heading_deg: float | None
    role: str = "memory"
    edge: str | None = None
    toward: str | None = None

def angle_diff(a: float, b: float) -> float:
    """a - b in (-180, 180]."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


@dataclass
class RefIndex:
    """A map with its reference embeddings, ready to match: row i of `vectors` is `refs[i]`."""

    pmap: PlaceMap
    vectors: np.ndarray  # [N, D] float32, unit rows
    refs: list  # ReferenceImage, aligned with vectors
    memory: MemoryIndex | None = None

    @property
    def model(self) -> str:
        """The place model live frames must be embedded with."""
        if self.pmap.embedding:
            return self.pmap.embedding.model
        return self.pmap.memory.model if self.pmap.memory else ""

    @property
    def empty(self) -> bool:
        return not self.refs and not (self.memory and self.memory.frames)

    @classmethod
    def build(cls, pmap: PlaceMap, embeddings: np.ndarray | None, exclude_rec: str | None = None,
              memory: tuple[list[MemoryFrame], np.ndarray] | None = None,
              memory_radius_m: float = MEMORY_RADIUS_M) -> "RefIndex":
        """Keep the references with an embedding row (hidden nodes are already out of a published map); without
        those taken from `exclude_rec` (replaying that walk against the others' references only). `memory`: the
        map's visual memory as nav/mapping/memory.load_memory gives it."""
        nodes = {n.id for n in pmap.nodes}
        refs, rows = [], []
        if embeddings is not None and embeddings.ndim == 2:
            for r in pmap.references:
                if r.embedding_row is not None and 0 <= r.embedding_row < len(embeddings) and r.node in nodes \
                        and r.rec_id != exclude_rec:
                    refs.append(r)
                    rows.append(r.embedding_row)
        dim = embeddings.shape[1] if embeddings is not None and embeddings.ndim == 2 else 0
        vectors = embeddings[rows].astype(np.float32) if rows else np.zeros((0, dim), np.float32)
        mem = MemoryIndex.build(pmap, *memory, radius_m=memory_radius_m, exclude_rec=exclude_rec) if memory else None
        return cls(pmap, vectors, refs, mem)

    def distances_from(self, node: str, unknown_m: float = 10.0) -> dict[str, float]:
        """Shortest walk (metres) from `node` to every node over the map's links (its connections before cleaning),
        either direction; through a merged area only in and out of the same part (as route.py)."""
        from nav.navigation.route import edge_parts

        areas = set(self.pmap.merged.values())
        graph: dict[str, list[tuple[str, float, str, str]]] = {}  # node -> (next node, metres, part left, reached)
        for e in self.pmap.links or self.pmap.edges:
            length = e.distance_m if e.distance_m is not None else unknown_m
            sp, tp = edge_parts(e)
            graph.setdefault(e.source, []).append((e.target, length, sp, tp))
            graph.setdefault(e.target, []).append((e.source, length, tp, sp))
        first = (node, "")
        best = {first: 0.0}
        queue = [(0.0, first)]
        while queue:
            d, state = heapq.heappop(queue)
            n, part = state
            if d > best.get(state, math.inf):
                continue
            for m, length, leave, reach in graph.get(n, []):
                if part and leave != part:
                    continue
                nxt = (m, reach if m in areas else "")
                if d + length < best.get(nxt, math.inf):
                    best[nxt] = d + length
                    heapq.heappush(queue, (d + length, nxt))
        out: dict[str, float] = {}
        for (n, _), d in best.items():
            out[n] = min(d, out.get(n, math.inf))
        return out

    def live_step_length(self) -> float | None:
        f = self.pmap.frame
        return (f.live_step_length_m or f.step_length_m) if f else None

    def neighbours(self, node: str) -> set[str]:
        out = {node}
        for e in self.pmap.links or self.pmap.edges:
            if e.source == node:
                out.add(e.target)
            elif e.target == node and e.bidirectional:
                out.add(e.source)
        return out


def camera_pitch(o) -> float | None:
    """Elevation of the rear-camera axis (device -z) from W3C Euler angles: 0 = level, -90 = at the floor."""
    if o is None:
        return None
    r = math.pi / 180
    return round(math.degrees(math.asin(max(-1.0, min(1.0, -math.cos(o.beta * r) * math.cos(o.gamma * r))))), 1)


def _usable_heading(m: LiveMotion | None) -> float | None:
    if m is None or m.heading_deg is None or m.heading_source is None:
        return None
    if m.heading_accuracy_deg is not None and m.heading_accuracy_deg > HEADING_MAX_ACCURACY_DEG:
        return None
    return m.heading_deg


def localize(index: RefIndex, queries: np.ndarray, thresholds: Thresholds, *, top_k: int = 5, refs_per_node: int = 3,
             previous: str | None = None, expected: str | None = None, heading_abs_deg: float | None = None,
             model: str = "", took_ms: dict[str, float] | None = None, motions: list[LiveMotion | None] | None = None,
             previous_step_count: int | None = None, verify=None,
             verify_params: VerifyParams | None = None) -> Localization:
    """`queries`: [F, D] unit place embeddings of the live frames (a burst of F >= 1), oldest first; `motions`: the
    phone's motion for each frame, aligned with `queries` (shorter or None: unknown). `verify(rows, frames)`: LightGlue
    inliers of the live frames (indexes into the burst) with those memory rows, {row: inliers}, or None."""
    pmap = index.pmap
    q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    names = {n.id: n.name for n in pmap.nodes}
    motions = list(motions or [])[:len(q)]
    motions += [None] * (len(q) - len(motions))
    newest = next((m for m in reversed(motions) if m is not None), None)

    # heading: the newest motion's compass (when trustworthy), else the explicit heading_deg
    live_heading = _usable_heading(newest)
    if live_heading is None:
        live_heading = heading_abs_deg
    heading_map = None
    if live_heading is not None and pmap.frame and pmap.frame.heading_zero_abs_deg is not None:
        heading_map = round((live_heading - pmap.frame.heading_zero_abs_deg) % 360, 1)

    # frames pointed at the floor or ceiling are left out
    pitches = [camera_pitch(m.orientation) if m else None for m in motions]
    keep = [i for i, p in enumerate(pitches) if p is None or abs(p) <= thresholds.max_abs_pitch_deg]

    # walked since the last confirmed node, from the phone's step count
    walked = None
    step_len = index.live_step_length()
    steps = newest.step_count if newest else None
    if previous is not None and previous_step_count is not None and steps is not None and step_len \
            and steps >= previous_step_count:  # a count below it: the phone restarted counting, no gate
        walked = round((steps - previous_step_count) * step_len, 2)

    base = dict(map_id=pmap.map_id, model=model, frames=len(q), previous=previous, expected=expected,
                heading_map_deg=heading_map, thresholds=thresholds, took_ms=took_ms or {},
                frames_used=len(keep), pitch_deg=pitches if any(p is not None for p in pitches) else [],
                walked_m=walked)
    if index.empty:
        return Localization(**base, status="lost", reason="the map has no reference images", candidates=[])
    if not keep:
        return Localization(**base, status="lost", reason="the camera points at the floor or ceiling: hold it up",
                            candidates=[])

    by_node: dict[str, list[tuple[float, object]]] = {}
    if index.refs:
        sims = (index.vectors @ q[keep].T).mean(axis=1)  # [N]
        for s, r in zip(sims.tolist(), index.refs):
            by_node.setdefault(r.node, []).append((s, r))
    position = None
    mem = index.memory
    verified: dict[int, int] = {}
    if mem and mem.frames:
        msims = (mem.vectors @ q[keep].T).mean(axis=1)  # [M]
        cos = msims
        if verify is not None:
            vp = verify_params or VerifyParams()
            got = verify(_diverse_top(msims, mem.frames, vp.k, vp.spread_ms), keep) if vp.k > 0 else None
            if got:
                verified = {int(i): int(k) for i, k in got.items()}
                msims = msims.copy()
                for i, k in verified.items():
                    msims[i] += vp.gain * vp.confidence(k)
        for i in np.argsort(-msims):
            node = mem.nodes[i]
            if node is not None and node in names:
                f = mem.frames[i]
                by_node.setdefault(node, []).append((float(msims[i]), _MemoryRef(f.id, f.image, f.heading_deg)))
        best_i = int(np.argmax(msims))
        f = mem.frames[best_i]
        if f.x_m is not None:
            position = LivePosition(x_m=f.x_m, y_m=f.y_m, heading_deg=f.heading_deg, frame=f.id, rec_id=f.rec_id,
                                    t_ms=f.t_ms, score=round(float(cos[best_i]), 4), node=mem.nodes[best_i],
                                    inliers=verified.get(best_i))
    base["position"] = position
    base["verified"] = {mem.frames[i].id: k for i, k in verified.items()} if verified else {}
    if not by_node:
        return Localization(**base, status="lost", reason="no reference or memory frame belongs to a node",
                            candidates=[])
    ranked = sorted(((max(s for s, _ in v), node) for node, v in by_node.items()), reverse=True)

    dist = index.distances_from(previous) if walked is not None and previous in names else None

    def plausible(node: str) -> bool | None:
        if dist is None:
            return None
        if node == previous:
            return True
        d = dist.get(node)
        if d is None:
            return False
        return walked >= d - max(thresholds.gate_slack_m, thresholds.gate_slack_frac * d)

    def node_match(score: float, node: str) -> NodeMatch:
        best_refs = sorted(by_node.get(node, []), key=lambda x: -x[0])[:refs_per_node]
        return NodeMatch(node=node, name=names.get(node, node), score=round(score, 4),
                         distance_m=None if dist is None or node not in dist else round(dist[node], 1),
                         plausible=plausible(node), refs=[
            RefMatch(id=r.id, score=round(s, 4), role=r.role, edge=r.edge, toward=r.toward, heading_deg=r.heading_deg,
                     heading_diff_deg=None if heading_map is None or r.heading_deg is None
                     else round(angle_diff(heading_map, r.heading_deg), 1),
                     image=r.image) for s, r in best_refs])

    best_score, best = ranked[0]
    margin = best_score - ranked[1][0] if len(ranked) > 1 else best_score
    if best_score < thresholds.lost_score:
        status, reason = "lost", f"best score {best_score:.2f} < {thresholds.lost_score:.2f}"
    elif best_score < thresholds.confirm_score:
        status, reason = "uncertain", f"best score {best_score:.2f} < {thresholds.confirm_score:.2f}"
    elif margin < thresholds.confirm_margin:
        status, reason = "uncertain", f"{names.get(ranked[1][1], ranked[1][1])} scores close (margin {margin:.2f})"
    elif dist is None and previous and previous in names and best not in index.neighbours(previous):
        status, reason = "uncertain", f"not next to the last confirmed node ({names[previous]})"
    elif plausible(best) is False:
        status, reason = "uncertain", (f"{names.get(best, best)} is not connected to {names[previous]}"
                                       if best not in dist else f"{names.get(best, best)} is {dist[best]:.0f} m from "
                                       f"{names[previous]}, only {walked:.0f} m walked")
    else:
        if dist is not None:
            where = f", {walked:.0f} m walked from the last confirmed node"
        else:
            where = ", next to the last confirmed node" if previous in names else ""
        status, reason = "confirmed", (f"clear: score >= {thresholds.confirm_score:.2f}, margin >= "
                                       f"{thresholds.confirm_margin:.2f}" + where)
    rank = next((i + 1 for i, (_, n) in enumerate(ranked) if n == expected), None)
    return Localization(**base, status=status, reason=reason, best=best, margin=round(margin, 4),
                        candidates=[node_match(s, n) for s, n in ranked[:top_k]],
                        expected_score=round(ranked[rank - 1][0], 4) if rank else None, expected_rank=rank)
