"""Routes on a place's map: shortest walk between two nodes over edges trusted enough, in the direction asked.

Edges are walkable source -> target when `status` is trusted, and target -> source when the edge is bidirectional
and `reverse_status` is trusted. Real users get `trust="verified"` (a person walked and checked it, review.py);
"observed" (walked while mapping) and "any" (also inferred) are for testing and for the mapper.
"""

import heapq
from dataclasses import dataclass, field

from nav.schemas.map import PlaceEdge, PlaceMap, Step

TRUSTED = {"verified": {"verified"}, "observed": {"verified", "observed"},
           "any": {"verified", "observed", "inferred"}}
UNKNOWN_LENGTH_M = 10.0  # cost of an edge without a distance


@dataclass
class Hop:
    edge: str
    source: str  # the node you leave
    target: str  # the node you reach
    forward: bool  # True: along the edge (source -> target); False: walking it back
    length_m: float | None
    bearing_deg: float | None  # walking direction when leaving `source`, map frame
    instruction: str | None
    steps: list[Step] = field(default_factory=list)
    status: str = "observed"
    arrival_deg: float | None = None  # walking direction when reaching `target`, map frame (arrival.py checks it)


def hops(e: PlaceEdge) -> list[Hop]:
    """Both directions of an edge (the reverse only if it is bidirectional)."""
    out = [Hop(e.id, e.source, e.target, True, e.distance_m, e.departure_heading_deg, e.instruction, e.steps,
               e.status, arrival_deg=e.arrival_heading_deg)]
    if e.bidirectional:
        back = None if e.arrival_heading_deg is None else round((e.arrival_heading_deg + 180) % 360, 1)
        back_arrival = None if e.departure_heading_deg is None else round((e.departure_heading_deg + 180) % 360, 1)
        out.append(Hop(e.id, e.target, e.source, False, e.distance_m, back, e.reverse_instruction,
                       e.reverse_steps, e.reverse_status or "inferred", arrival_deg=back_arrival))
    return out


def plan_route(pmap: PlaceMap, start: str, goal: str, trust: str = "verified") -> list[Hop] | None:
    """The shortest route by distance (Dijkstra) from `start` to `goal`, or None if trusted edges don't connect
    them. [] when start == goal."""
    allowed = TRUSTED[trust]
    graph: dict[str, list[Hop]] = {}
    for e in pmap.edges:
        for h in hops(e):
            if h.status in allowed:
                graph.setdefault(h.source, []).append(h)
    best = {start: 0.0}
    came: dict[str, Hop] = {}
    queue = [(0.0, start)]
    while queue:
        d, node = heapq.heappop(queue)
        if node == goal:
            route = []
            while node != start:
                route.append(came[node])
                node = came[node].source
            return route[::-1]
        if d > best.get(node, float("inf")):
            continue
        for h in graph.get(node, []):
            nd = d + (h.length_m if h.length_m is not None else UNKNOWN_LENGTH_M)
            if nd < best.get(h.target, float("inf")):
                best[h.target], came[h.target] = nd, h
                heapq.heappush(queue, (nd, h.target))
    return None
