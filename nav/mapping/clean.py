"""Cleaning: the published map keeps only what the walks back up (less is more).

The LLM's `confidence` does not follow the evidence (Itnig: a "high" bathroom seen once by one walk), so code decides
from the walks, after the review (review.py) and before publishing:

1. Inferred edges (no walk went directly between their nodes) go.
2. One edge per pair of spots: the strongest (verified, walks, confidence, shorter).
3. A weak spot on the way (seen by fewer than `min_node_walks` walks, not protected, exactly two edges) is folded
   into one edge between its neighbours: steps chained, distances added, `via` lists it.
4. Edges walked by at least `min_edge_walks` walks (either way) are the core. Weaker edges come back only where the
   map would otherwise fall apart: the strongest one joining two parts (Kruskal), flagged for review.
5. A core edge longer than another kept way between its ends goes.
6. Spots left without edges go, unless protected; step 3 runs again on what is left.

Protected, never pruned: entrances, destinations and rooms, verified items, and anything the review touched (`keep`
rescues an item; names, approvals, merges count too). What is pruned is listed in `PlaceMap.pruned` with the reason;
the build's own map.json keeps everything, and the next build starts from it (build.py), so a spot more walks back
up comes back with its id.
"""

from __future__ import annotations

from pydantic import BaseModel

from nav.mapping.review import NodeReview, Review
from nav.schemas.map import CleanInfo, ItemQuality, PlaceEdge, PlaceMap, Status

RANK: dict[str, int] = {"inferred": 0, "observed": 1, "verified": 2}
CONF = {"low": 0, "medium": 1, "high": 2}


class CleanParams(BaseModel):
    min_edge_walks: int = 2  # walks (either direction) that make an edge confident
    min_node_walks: int = 2  # walks whose frames show a spot that make it confident
    protect_kinds: list[str] = ["entrance", "destination", "room"]
    connect: bool = True  # keep the strongest weak edge where the confident ones leave the map in pieces
    contract: bool = True  # fold weak spots on the way into one edge


def edge_walks(e: PlaceEdge) -> int:
    if e.walked_forward is None and e.walked_back is None:  # built before walks were counted
        return 1 if e.walked else 0
    return (e.walked_forward or 0) + (e.walked_back or 0)


def node_walks(n) -> int:
    return len({ev.rec_id for ev in n.evidence})


def _verified(e: PlaceEdge) -> bool:
    return e.status == "verified" or e.reverse_status == "verified"


def _score(e: PlaceEdge) -> tuple:
    return (_verified(e), edge_walks(e), CONF.get(e.confidence, 0), -(e.distance_m if e.distance_m is not None else 1e9))


def _ends(e: PlaceEdge) -> frozenset:
    """The spots an edge joins: two edges to different parts of a merged area are not duplicates."""
    return frozenset((e.source_part or e.source, e.target_part or e.target))


def _flipped(e: PlaceEdge) -> PlaceEdge:
    """A two-way edge written the other way round."""
    def back(h: float | None) -> float | None:
        return None if h is None else round((h + 180) % 360, 1)

    return e.model_copy(deep=True, update={
        "source": e.target, "target": e.source, "source_part": e.target_part, "target_part": e.source_part,
        "instruction": e.reverse_instruction or e.instruction, "reverse_instruction": e.instruction,
        "steps": e.reverse_steps, "reverse_steps": e.steps, "candidates": e.candidates[::-1],
        "path": e.path[::-1], "status": e.reverse_status or "inferred", "reverse_status": e.status,
        "walked_forward": e.walked_back, "walked_back": e.walked_forward,
        "departure_heading_deg": back(e.arrival_heading_deg), "arrival_heading_deg": back(e.departure_heading_deg),
        "turn_deg": None, "instruction_kind": None})


def _weakest(*s: Status | None) -> Status | None:
    if any(x is None for x in s):
        return None
    return min(s, key=RANK.__getitem__)


def _then(a: str | None, b: str | None) -> str | None:
    if not a or not b:
        return a or b
    return f"{a.rstrip()} Then, {b[0].lower()}{b[1:]}"


def _union(a: list, b: list) -> list:
    return list(dict.fromkeys([*a, *b]))


def _quality(a: ItemQuality | None, b: ItemQuality | None) -> ItemQuality | None:
    if a is None or b is None:
        return a or b
    return ItemQuality(visual_p=max(a.visual_p, b.visual_p), usable=min(a.usable, b.usable),
                       turn_agrees_p=a.turn_agrees_p, flags=_union(a.flags, b.flags))


def _parts(nodes, edges: list[PlaceEdge]) -> int:
    """How many strongly connected parts the spots make, over the directions that were walked (or verified)."""
    walked = {"observed", "verified"}
    out: dict[str, list[str]] = {n: [] for n in nodes}
    for e in edges:
        if e.status in walked:
            out[e.source].append(e.target)
        if e.bidirectional and e.reverse_status in walked:
            out[e.target].append(e.source)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    count = 0

    def visit(v: str) -> None:  # Tarjan, small graphs
        nonlocal count
        index[v] = low[v] = len(index)
        stack.append(v)
        on.add(v)
        for w in out[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            count += 1
            while (w := stack.pop()) != v:
                on.discard(w)
            on.discard(v)

    for v in out:
        if v not in index:
            visit(v)
    return count


def _join(a: PlaceEdge, b: PlaceEdge, via: str, name: str) -> PlaceEdge:
    """a: X -> via, b: via -> Y, as one edge X -> Y (a's id)."""
    def add(x: float | None, y: float | None) -> float | None:
        return None if x is None or y is None else round(x + y, 1)

    def least(x: int | None, y: int | None) -> int | None:
        return None if x is None and y is None else min(x or 0, y or 0)

    return a.model_copy(deep=True, update={
        "target": b.target, "target_part": b.target_part,
        "kind": a.kind if a.kind == b.kind or b.kind == "walk" else b.kind,
        "bidirectional": a.bidirectional and b.bidirectional,
        "steps": [*a.steps, *b.steps], "reverse_steps": [*b.reverse_steps, *a.reverse_steps],
        "instruction": _then(a.instruction, b.instruction),
        "reverse_instruction": _then(b.reverse_instruction, a.reverse_instruction),
        "landmarks": _union(a.landmarks, b.landmarks), "hazards": _union(a.hazards, b.hazards),
        "visual_cues": _union(a.visual_cues, b.visual_cues), "candidates": _union(a.candidates, b.candidates),
        "confidence": min(a.confidence, b.confidence, key=CONF.__getitem__),
        "distance_m": add(a.distance_m, b.distance_m), "measured_distance_m": add(a.measured_distance_m,
                                                                                  b.measured_distance_m),
        "path": [*a.path, *b.path], "walked": a.walked and b.walked,
        "arrival_heading_deg": b.arrival_heading_deg, "quality": _quality(a.quality, b.quality),
        "status": _weakest(a.status, b.status), "reverse_status": _weakest(a.reverse_status, b.reverse_status),
        "walked_forward": least(a.walked_forward, b.walked_forward), "walked_back": least(a.walked_back, b.walked_back),
        "review_flags": _union([*a.review_flags, *b.review_flags], [f"goes past {name} (folded in: few walks saw it)"]),
        "via": [*a.via, via, *b.via]})


class _Cleaner:
    def __init__(self, m: PlaceMap, review: Review, p: CleanParams):
        self.m, self.p = m, p
        self.pruned: dict[str, str] = {}
        self.folded: dict[str, str] = {}  # edge id -> the edge it was folded into
        self.nodes = {n.id: n for n in m.nodes}
        self.names = {n.id: n.name for n in m.nodes}
        blank = NodeReview().model_dump(exclude={"note"})
        reviewed = {i for i, r in review.nodes.items() if r.model_dump(exclude={"note"}) != blank}
        self.protected = {n.id for n in m.nodes if n.kind in p.protect_kinds or n.status == "verified"
                          or n.id in reviewed or n.id in set(m.merged.values())}
        self.strong = self.protected | {n.id for n in m.nodes if node_walks(n) >= p.min_node_walks}
        self.kept_edges = {i for i, r in review.edges.items() if r.keep}
        self.edges = list(m.edges)

    def safe(self, e: PlaceEdge) -> bool:
        return _verified(e) or e.id in self.kept_edges

    def drop_inferred(self) -> None:
        keep = []
        for e in self.edges:
            if not e.walked and not self.safe(e):
                self.pruned[e.id] = "inferred: no walk went directly between its spots"
            else:
                keep.append(e)
        self.edges = keep

    def dedupe(self) -> None:
        best: dict[frozenset, PlaceEdge] = {}
        for e in self.edges:
            k = _ends(e)
            if k not in best:
                best[k] = e
                continue
            win, lose = sorted([best[k], e], key=lambda x: (self.safe(x), *_score(x)), reverse=True)
            best[k] = win
            if self.safe(lose):  # two verified ways between the same spots: both stay
                best[frozenset((lose.id,))] = lose
            else:
                self.pruned[lose.id] = f"same spots as {win.id}, which is stronger"
        self.edges = list(best.values())

    def incident(self, nid: str) -> list[PlaceEdge]:
        return [e for e in self.edges if nid in (e.source, e.target)]

    def contract(self) -> bool:
        changed = False
        for nid in list(self.nodes):
            if nid in self.strong:
                continue
            inc = self.incident(nid)
            if len(inc) != 2 or any(self.safe(e) for e in inc):
                continue
            a, b = inc
            if a.source == a.target or b.source == b.target or {a.source, a.target} == {b.source, b.target}:
                continue
            if any((e.source == nid and e.source_part) or (e.target == nid and e.target_part) for e in inc):
                continue
            if a.target != nid:
                if not a.bidirectional:
                    continue
                a = _flipped(a)
            if b.source != nid:
                if not b.bidirectional:
                    continue
                b = _flipped(b)
            joined = _join(a, b, nid, self.names[nid])
            self.edges = [e for e in self.edges if e.id not in (a.id, b.id)] + [joined]
            self.pruned[nid] = f"seen by {node_walks(self.nodes[nid])} walk(s): folded into {joined.id}"
            self.pruned[b.id] = f"folded into {joined.id} (through {self.names[nid]})"
            self.folded[b.id] = joined.id
            self.drop_node(nid)
            self.dedupe()
            changed = True
        return changed

    def drop_node(self, nid: str) -> None:
        del self.nodes[nid]

    def connect(self) -> None:
        core = [e for e in self.edges if self.safe(e) or edge_walks(e) >= self.p.min_edge_walks]
        rest = sorted((e for e in self.edges if e not in core), key=_score, reverse=True)
        parent = {i: i for i in self.nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in core:
            parent[find(e.source)] = find(e.target)
        keep = list(core)
        for e in rest:
            a, b = find(e.source), find(e.target)
            if self.p.connect and a != b:
                parent[a] = b
                n = edge_walks(e)
                e.review_flags = _union(e.review_flags, [f"kept to connect the map: walked {n} time{'s' * (n != 1)}"])
                keep.append(e)
            else:
                self.pruned[e.id] = (f"walked {edge_walks(e)} time(s): its spots are already joined by stronger edges"
                                     if a == b else f"walked {edge_walks(e)} time(s)")
        if self.p.connect:
            self.walkable(keep, [e for e in rest if e not in keep])
        self.edges = keep

    def walkable(self, keep: list[PlaceEdge], rest: list[PlaceEdge]) -> None:
        """Routes go only the way edges were walked (route.py, trust observed): bring back the strongest weak
        edges that let more spots reach each other that way (fewer strongly connected parts)."""
        before = _parts(self.nodes, keep)
        for e in rest:
            after = _parts(self.nodes, [*keep, e])
            if after < before:
                n = edge_walks(e)
                e.review_flags = _union(e.review_flags, [f"kept so the walked way reaches every spot: walked {n} "
                                                         f"time{'s' * (n != 1)}"])
                keep.append(e)
                self.pruned.pop(e.id, None)
                before = after

    def shortcuts(self) -> None:
        """A core edge no shorter than another kept way between its ends, in each direction it was walked, goes."""
        from nav.navigation.route import TRUSTED, plan_route

        for e in sorted(self.edges, key=_score):
            if self.safe(e) or e.distance_m is None:
                continue
            others = self.m.model_copy(update={"edges": [x for x in self.edges if x.id != e.id],
                                               "nodes": list(self.nodes.values())})
            ways = [(e.source, e.target, e.status), (e.target, e.source, e.reverse_status if e.bidirectional else None)]
            arounds = []
            for a, b, status in ways:
                if status not in TRUSTED["observed"]:
                    continue
                route = plan_route(others, a, b, trust="observed")
                around = sum(h.length_m for h in route) if route and all(h.length_m is not None for h in route) \
                    else None
                arounds.append(around)
            if arounds and all(x is not None and x <= e.distance_m for x in arounds):
                self.edges = [x for x in self.edges if x.id != e.id]
                self.pruned[e.id] = (f"{e.distance_m:g} m, no shorter than the {max(arounds):.1f} m walked way "
                                     "through other spots")

    def trim(self) -> None:
        used = {x for e in self.edges for x in (e.source, e.target)}
        for nid in list(self.nodes):
            if nid in used:
                continue
            if nid in self.protected:
                self.m.warnings.append(f"clean: {nid} ({self.names[nid]}) has no confident connection")
            else:
                self.pruned[nid] = f"seen by {node_walks(self.nodes[nid])} walk(s), no confident connection"
                self.drop_node(nid)

    def run(self) -> PlaceMap:
        m, before = self.m, (len(self.m.nodes), len(self.m.edges))
        self.drop_inferred()
        self.dedupe()
        while self.p.contract and self.contract():
            pass
        self.connect()
        self.shortcuts()
        self.trim()
        while self.p.contract and self.contract():
            pass
        self.finish()
        m.clean = CleanInfo(params=self.p.model_dump(), nodes_before=before[0], edges_before=before[1],
                            nodes=len(m.nodes), edges=len(m.edges))
        return m

    def finish(self) -> None:
        m = self.m
        gone = {i for i in self.names if i not in self.nodes}
        gone_names = {self.names[i].casefold() for i in gone}
        order = {e.id: k for k, e in enumerate(m.edges)}
        m.nodes = [n for n in m.nodes if n.id in self.nodes]
        m.edges = sorted(self.edges, key=lambda e: order.get(e.id, len(order)))
        for n in m.nodes:
            n.relations = [r for r in n.relations if r.subject.casefold() not in gone_names]
        targets = {e.id: e.target for e in m.edges}
        sources = {e.id: e.source for e in m.edges}
        refs = []
        for r in m.references:
            while r.edge in self.folded:
                r.edge = self.folded[r.edge]
            if r.node in gone:
                continue
            if r.edge is not None and r.edge not in targets:  # leaving along a pruned edge: a view of the spot
                r.role, r.edge, r.toward = "at", None, None
            elif r.edge is not None:  # an edge folded through a spot now leads further
                r.toward = targets[r.edge] if sources[r.edge] == r.node else sources[r.edge]
            refs.append(r)
        m.references = refs
        ids = {r.id for r in refs}
        for n in m.nodes:
            n.references = [x for x in n.references if x in ids]
        for e in m.edges:
            e.reference_views = [x for x in e.reference_views if x in ids]
        m.pruned = dict(sorted(self.pruned.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]) if kv[0][1:].isdigit()
                                                                    else 0)))


def clean_map(pmap: PlaceMap, review: Review | None = None, params: CleanParams | None = None) -> PlaceMap:
    """The map with only confident nodes and edges (a copy; see the module docstring)."""
    return _Cleaner(pmap.model_copy(deep=True), review or Review(), params or CleanParams()).run()

