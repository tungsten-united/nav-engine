"""Visual place recognition against a map's reference images (docs/navigation_stack.md 4.1 and 6): live frames ->
ranked nodes and a confidence status. Stateless: agreement across consecutive results (4.2) is the caller's (Orient's orchestrator).

Each live frame is a place embedding (the map's `embedding.model`, MegaLoc), compared by cosine with every reference
image of the map (`embeddings.npy`, unit rows). A burst of frames scores each reference by their mean similarity. A
node scores as its best reference; the margin is the best node's score minus the second's.

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

from nav.schemas.map import PlaceMap
from nav.schemas.navigation import LiveMotion, Localization, NodeMatch, RefMatch, Thresholds

HEADING_MAX_ACCURACY_DEG = 45.0  # a compass reporting a worse accuracy is not used

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

    @classmethod
    def build(cls, pmap: PlaceMap, embeddings: np.ndarray | None) -> "RefIndex":
        """Keep the references with an embedding row (hidden nodes are already out of a published map)."""
        nodes = {n.id for n in pmap.nodes}
        refs, rows = [], []
        if embeddings is not None and embeddings.ndim == 2:
            for r in pmap.references:
                if r.embedding_row is not None and 0 <= r.embedding_row < len(embeddings) and r.node in nodes:
                    refs.append(r)
                    rows.append(r.embedding_row)
        dim = embeddings.shape[1] if embeddings is not None and embeddings.ndim == 2 else 0
        vectors = embeddings[rows].astype(np.float32) if rows else np.zeros((0, dim), np.float32)
        return cls(pmap, vectors, refs)

    def distances_from(self, node: str, unknown_m: float = 10.0) -> dict[str, float]:
        """Shortest walk (metres) from `node` to every node over the map's edges, either direction."""
        graph: dict[str, list[tuple[str, float]]] = {}
        for e in self.pmap.edges:
            length = e.distance_m if e.distance_m is not None else unknown_m
            graph.setdefault(e.source, []).append((e.target, length))
            graph.setdefault(e.target, []).append((e.source, length))
        best = {node: 0.0}
        queue = [(0.0, node)]
        while queue:
            d, n = heapq.heappop(queue)
            if d > best.get(n, math.inf):
                continue
            for m, length in graph.get(n, []):
                if d + length < best.get(m, math.inf):
                    best[m] = d + length
                    heapq.heappush(queue, (d + length, m))
        return best

    def live_step_length(self) -> float | None:
        f = self.pmap.frame
        return (f.live_step_length_m or f.step_length_m) if f else None

    def neighbours(self, node: str) -> set[str]:
        out = {node}
        for e in self.pmap.edges:
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
             previous_step_count: int | None = None) -> Localization:
    """`queries`: [F, D] unit place embeddings of the live frames (a burst of F >= 1), oldest first; `motions`: the
    phone's motion for each frame, aligned with `queries` (shorter or None: unknown)."""
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
    if not index.refs:
        return Localization(**base, status="lost", reason="the map has no reference images", candidates=[])
    if not keep:
        return Localization(**base, status="lost", reason="the camera points at the floor or ceiling: hold it up",
                            candidates=[])

    sims = (index.vectors @ q[keep].T).mean(axis=1)  # [N]
    by_node: dict[str, list[tuple[float, object]]] = {}
    for s, r in zip(sims.tolist(), index.refs):
        by_node.setdefault(r.node, []).append((s, r))
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
        best_refs = sorted(by_node[node], key=lambda x: -x[0])[:refs_per_node]
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
