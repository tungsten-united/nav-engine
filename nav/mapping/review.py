"""Human review of a place's map (docs/navigation_stack.md 3, step 5): what a person checked on site.

The review is an overlay, `maps/<map_id>/review.json`, keyed by the map's stable ids (continuity.py), so approvals
and corrections survive rebuilds: a rebuild that keeps a node keeps its review too. Applying it to a build's map
gives the published map.json:

- approve: status -> verified. Routing for real users only uses verified edges (nav/navigation/route.py).
  Edges are approved per direction: `approve` for source -> target, `approve_reverse` for target -> source.
- name / kind / description / instruction / reverse_instruction: corrections that win over what the LLM wrote.
- merge_into: the node is part of another one (one big area the LLM split, say). Its reference images, evidence,
  relations and cues go to that node, its edges are joined to it (an edge between two parts is dropped), and
  `merged` lists it. Each joined edge end remembers the part it was walked to (`source_part` / `target_part`): its
  distance is to that part. Routes and the walked-distance gate pass through an area only in and out of the same
  part (route.py, localize.py): the walk from one part to another isn't in the map. Parts that a walk of at most
  JOIN_PARTS_M joined (the dropped edge between them) count as one part.
- keep: publish the item even where the cleaner (clean.py) would leave it out for too few walks.
- hide: leave the item out of the published map (it stays in the build's map.json; nothing is deleted).
  Hiding a node hides its edges, relations and reference images too.
- distances: metres measured on site between two nodes (tape, pacing, a floor plan). The walked metres come from
  the floor odometry with an assumed phone height (odometry.py); one scale factor, least squares over all
  measurements, rescales every metric quantity of the published map (edge and step distances, positions, paths,
  tracks, step length) and each measurement's residual is reported in `frame.measured_distances`.

Review entries whose ids are not in the map (retired, or from a later build) are kept and reported.

`seen` holds the name each node the review refers to had in the build (before the review) when the review was saved:
its entries' nodes, merge targets, the ends of its edges and measured distances. Saving fills in the names missing
from it. A rebuild that puts another place under one of those ids while the reviewed place is under another id now
(its name is there) holds every entry that depends on it: not applied, reported with where the place is now (Itnig,
2026-10-13: the first metric-1 build gave the stone wall tables' id to the judges' table, and the merge decided on
site moved the judges' table into the centre tables, 6 m away). A place the LLM only renamed keeps its entries.
Moving the entries to the new id, or deleting the `seen` name to confirm the place now under the id, lets them apply.

Adapted from tungsten-united/demo-repository@c5dc26f (navigation, mapping/src/mapper/review.py) by LPA.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from nav.schemas.map import NodeKind, PlaceMap, Point


class NodeReview(BaseModel):
    approve: bool = False
    hide: bool = False
    keep: bool = False  # publish it even if the cleaner (clean.py) would leave it out
    name: str | None = None
    kind: NodeKind | None = None
    description: str | None = None
    merge_into: str | None = None  # node id this node is part of (see the module docstring)
    note: str | None = None


class EdgeReview(BaseModel):
    approve: bool = False  # source -> target walked and checked
    approve_reverse: bool = False  # target -> source walked and checked
    hide: bool = False
    keep: bool = False  # publish it even if the cleaner (clean.py) would leave it out
    instruction: str | None = None
    reverse_instruction: str | None = None
    note: str | None = None


class RelationReview(BaseModel):
    approve: bool = False
    hide: bool = False


class MeasuredDistance(BaseModel):
    between: tuple[str, str]  # node ids; the distance is along the shortest walk between them
    meters: float
    sigma_m: float = 0.5  # how sure the measurement is
    note: str | None = None  # how it was measured


class Review(BaseModel):
    updated_at_utc: datetime | None = None
    updated_by: str | None = None
    nodes: dict[str, NodeReview] = {}
    edges: dict[str, EdgeReview] = {}
    relations: dict[str, RelationReview] = {}
    distances: list[MeasuredDistance] = []
    seen: dict[str, str] = {}  # node id -> its name when the review was saved (see the module docstring)


def review_path(map_dir: Path) -> Path:
    return map_dir / "review.json"


def load_review(map_dir: Path) -> Review:
    path = review_path(map_dir)
    return Review.model_validate_json(path.read_text()) if path.is_file() else Review()


def _build_map(map_dir: Path) -> PlaceMap | None:
    """The published build's own map (before the review), as build.build_map_of."""
    published = map_dir / "map.json"
    if not published.is_file():
        return None
    path = map_dir / "builds" / PlaceMap.model_validate_json(published.read_text()).build.build_id / "map.json"
    return PlaceMap.model_validate_json(path.read_text()) if path.is_file() else None


def referenced(pmap: PlaceMap, review: Review) -> set[str]:
    """The node ids the review's entries depend on: their nodes, merge targets, edge ends and measured distances."""
    ends = {e.id: (e.source, e.target) for e in pmap.edges}
    out = set(review.nodes) | {r.merge_into for r in review.nodes.values() if r.merge_into}
    out |= {x for i in review.edges for x in ends.get(i, ())}
    return out | {x for d in review.distances for x in d.between}


def with_seen(pmap: PlaceMap | None, review: Review) -> Review:
    """The review with `seen` filled in from the build's map (`pmap`, before the review): names already there stay,
    names of nodes it no longer refers to go."""
    if pmap is None:
        return review
    names = {n.id: n.name for n in pmap.nodes}
    refs = referenced(pmap, review)
    seen = {i: n for i, n in review.seen.items() if i in refs}
    seen |= {i: names[i] for i in sorted(refs) if i not in seen and i in names}
    return review.model_copy(update={"seen": seen})


def save_review(map_dir: Path, review: Review, by: str | None = None) -> Review:
    review = with_seen(_build_map(map_dir), review)
    review = review.model_copy(update={"updated_at_utc": datetime.now(timezone.utc), "updated_by": by})
    tmp = map_dir / ".review.json.tmp"
    tmp.write_text(review.model_dump_json(indent=1, exclude_defaults=False))
    tmp.replace(review_path(map_dir))
    return review


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()


def moved(pmap: PlaceMap, review: Review) -> dict[str, str]:
    """Node id -> the id the place reviewed under it has in this build: another place has the id now, and the
    reviewed one's name (unique in the build) is elsewhere. A place renamed (its old name gone) has not moved."""
    names = {n.id: n.name for n in pmap.nodes}
    counts: dict[str, int] = {}
    for n in names.values():
        counts[_key(n)] = counts.get(_key(n), 0) + 1
    where = {_key(n): i for i, n in names.items() if counts[_key(n)] == 1}
    return {i: where[_key(was)] for i, was in review.seen.items()
            if i in names and _key(names[i]) != _key(was) and where.get(_key(was), i) != i}


def _hold(pmap: PlaceMap, review: Review, gone: dict[str, str], warnings: list[str]) -> Review:
    """The review without the entries that depend on a node whose place moved (`moved`), each reported."""
    names = {n.id: n.name for n in pmap.nodes}
    ends = {e.id: (e.source, e.target) for e in pmap.edges}
    for i, j in sorted(gone.items()):
        warnings.append(f"review: {i} was {review.seen[i]!r} when reviewed; this build has {names[i]!r} there and "
                        f"{review.seen[i]!r} at {j}: entries that depend on {i} are held (move them to {j}, or delete "
                        f"seen.{i} to keep them on {names[i]!r})")
    nodes = {i: r for i, r in review.nodes.items() if i not in gone and r.merge_into not in gone}
    edges = {i: r for i, r in review.edges.items() if not set(ends.get(i, ())) & set(gone)}
    distances = [d for d in review.distances if not set(d.between) & set(gone)]
    return review.model_copy(update={"nodes": nodes, "edges": edges, "distances": distances})


def apply_review(pmap: PlaceMap, review: Review) -> PlaceMap:
    """The map as published: the build's map with the review applied (a copy; the input is unchanged)."""
    m = pmap.model_copy(deep=True)
    if review == Review():
        return m
    original = review
    if gone := moved(pmap, review):
        review = _hold(pmap, review, gone, m.warnings)
    hidden: set[str] = set()
    merged = _merges(m, review)
    if merged:
        _merge(m, merged, hidden)
    for n in m.nodes:
        r = review.nodes.get(n.id)
        if r is None:
            continue
        if r.hide:
            hidden.add(n.id)
            continue
        if r.name and r.name != n.name:
            n.name = r.name
            n.review_flags = [f for f in n.review_flags if f != "name set in review"] + ["name set in review"]
        if r.kind:
            n.kind = r.kind
        if r.description:
            n.description = r.description
        if r.approve:
            n.status = "verified"
            n.review_flags = [f for f in n.review_flags if f == "name set in review"]
    for n in m.nodes:
        for rel in n.relations:
            r = review.relations.get(rel.id)
            if n.id in hidden or (r and r.hide):
                hidden.add(rel.id)
            elif r and r.approve:
                rel.status = "verified"
        n.relations = [rel for rel in n.relations if rel.id not in hidden]
    for e in m.edges:
        r = review.edges.get(e.id)
        if e.source in hidden or e.target in hidden or (r and r.hide):
            hidden.add(e.id)
            continue
        if r is None:
            continue
        if r.instruction:
            e.instruction = r.instruction
        if r.reverse_instruction:
            e.reverse_instruction = r.reverse_instruction
        if r.approve:
            e.status = "verified"
            e.review_flags = [f for f in e.review_flags if f == "the way back was not walked"]
        if r.approve_reverse and e.bidirectional:
            e.reverse_status = "verified"
            e.review_flags = [f for f in e.review_flags if f != "the way back was not walked"]
    m.nodes = [n for n in m.nodes if n.id not in hidden]
    m.edges = [e for e in m.edges if e.id not in hidden]
    m.references = [x for x in m.references if x.node not in hidden]
    for e in m.edges:
        e.reference_views = [x for x in e.reference_views if any(r.id == x for r in m.references)]
    if review.distances:
        _scale_distances(m, review.distances)
    m.hidden = sorted(hidden)
    m.reviewed_at_utc = review.updated_at_utc or datetime.now(timezone.utc)
    known = {n.id for n in pmap.nodes} | {e.id for e in pmap.edges} | {r.id for n in pmap.nodes for r in n.relations}
    stale = sorted(i for i in [*original.nodes, *original.edges, *original.relations] if i not in known)
    if stale:
        m.warnings.append(f"review: entries for ids not in this build (kept for later): {', '.join(stale)}")
    return m


JOIN_PARTS_M = 3.0  # parts of a merged area a walk this short joined count as one part for routes


def _merges(m: PlaceMap, review: Review) -> dict[str, str]:
    """Node id -> the node it is merged into, chains followed (a -> b -> c: a -> c). Merges into a node that is
    missing or hidden, cycles and hidden nodes are left out, with a warning."""
    ids = {n.id for n in m.nodes}
    hidden = {i for i, r in review.nodes.items() if r.hide}
    out: dict[str, str] = {}
    for nid, r in review.nodes.items():
        if not r.merge_into or nid not in ids or nid in hidden:
            continue
        seen, target = [nid], r.merge_into
        while target not in seen and (nxt := review.nodes.get(target)) and nxt.merge_into:
            seen.append(target)
            target = nxt.merge_into
        if target in seen:
            m.warnings.append(f"review: node {nid}: merge cycle {' -> '.join(seen + [target])}, not merged")
        elif target not in ids or target in hidden:
            m.warnings.append(f"review: node {nid}: merge into {target}, which is not in this build or is hidden, "
                              "not merged")
        else:
            out[nid] = target
    return out


def _merge(m: PlaceMap, merged: dict[str, str], hidden: set[str]) -> None:
    """Fold each merged node into its node, in place (see the module docstring). Dropped edges go to `hidden`."""
    nodes = {n.id: n for n in m.nodes}
    for old, new in merged.items():
        a, b = nodes[new], nodes[old]
        for f in ("landmarks", "hazards", "visual_cues", "candidates", "references"):
            setattr(a, f, list(dict.fromkeys([*getattr(a, f), *getattr(b, f)])))
        a.evidence = [*a.evidence, *b.evidence]
        a.relations = [*a.relations, *b.relations]
    old_names = {nodes[x].name.casefold() for x in merged} | {nodes[x].name.casefold() for x in set(merged.values())}
    for new in set(merged.values()):  # one relation per subject, none about the area itself
        a, seen, keep = nodes[new], set(), []
        for rel in a.relations:
            key = rel.subject.casefold()
            if key not in seen and key not in old_names:
                seen.add(key)
                keep.append(rel)
        a.relations = keep
    m.nodes = [n for n in m.nodes if n.id not in merged]

    loops = {e.id: e for e in m.edges if merged.get(e.source, e.source) == merged.get(e.target, e.target)}
    # Parts a short walk inside the area joined count as one part (the error is under JOIN_PARTS_M).
    group = {x: x for x in [*merged, *merged.values()]}

    def find(x: str) -> str:
        while group[x] != x:
            x = group[x]
        return x

    for e in loops.values():
        if e.distance_m is not None and e.distance_m <= JOIN_PARTS_M:
            a, b = sorted((find(e.source), find(e.target)), key=lambda x: (x not in merged.values(), x))
            group[b] = a  # the area's own node, if in the group, stands for it
    edges = []
    for e in m.edges:
        if e.id in loops:  # between two parts of one area: the walk inside it
            continue
        for end in ("source", "target"):
            x = getattr(e, end)
            node = merged.get(x, x)
            if node != x or x in group:
                part = find(x)
                setattr(e, end, node)
                setattr(e, f"{end}_part", None if part == node else part)
        edges.append(e)
    m.edges = edges
    hidden |= set(loops)
    for r in m.references:
        r.node = merged.get(r.node, r.node)
        if r.toward is not None:
            r.toward = merged.get(r.toward, r.toward)
        if r.edge in loops:  # a view leaving one part toward another: a view of the area
            r.role, r.edge, r.toward = "at", None, None
    m.merged = dict(merged)


def _walk_m(pmap: PlaceMap, a: str, b: str) -> float | None:
    from nav.navigation.route import plan_route

    route = plan_route(pmap, a, b, trust="any")
    if route is None or any(h.length_m is None for h in route):
        return None
    return sum(h.length_m for h in route)


def _scale_distances(m: PlaceMap, measured: list[MeasuredDistance]) -> float:
    """Fit one scale factor to the measured distances (weighted least squares) and apply it in place."""
    pairs = [(d, _walk_m(m, *d.between)) for d in measured]
    usable = [(d, x) for d, x in pairs if x]
    for d, x in pairs:
        if not x:
            m.warnings.append(f"review: measured distance {d.between[0]} - {d.between[1]}: no walked connection "
                              "with distances in this build, not used")
    k = 1.0
    if usable:
        k = (sum(x * d.meters / d.sigma_m ** 2 for d, x in usable)
             / sum(x * x / d.sigma_m ** 2 for d, x in usable))

    def s(v: float | None, nd: int = 2) -> float | None:
        return None if v is None else round(v * k, nd)

    for e in m.edges:
        e.distance_m, e.measured_distance_m = s(e.distance_m, 1), s(e.measured_distance_m, 1)
        for st in [*e.steps, *e.reverse_steps]:
            st.distance_m = s(st.distance_m, 1)
        # New points: a node's position and its edges' path ends can be the same object.
        e.path = [Point(x_m=s(p.x_m), y_m=s(p.y_m)) for p in e.path]
    for n in m.nodes:
        if n.position:
            n.position = Point(x_m=s(n.position.x_m, 1), y_m=s(n.position.y_m, 1))
    for tr in m.tracks:
        tr.points = [(t, s(x), s(y)) for t, x, y in tr.points]
    if m.frame:
        from nav.schemas.map import MeasuredDistanceCheck

        m.frame.step_length_m = s(m.frame.step_length_m, 3)
        m.frame.live_step_length_m = s(m.frame.live_step_length_m, 3)
        m.frame.distance_scale = round(m.frame.distance_scale * k, 4)
        m.frame.measured_distances = []
        for d, x in pairs:
            after = None if not x else round(x * k, 2)
            m.frame.measured_distances.append(MeasuredDistanceCheck(
                between=d.between, meters=d.meters, sigma_m=d.sigma_m, map_m=after,
                residual_m=None if after is None else round(after - d.meters, 2)))
    return k


def template(pmap: PlaceMap, review: Review) -> dict:
    """An editable review.json for `nav map review`: every item with what to check, current entries filled in."""
    names = {n.id: n.name for n in pmap.nodes}

    def flags(item, kind: str) -> list[str]:
        q = item.quality.flags if item.quality else []
        warned = [w for w in pmap.warnings if f"{kind} {item.id}:" in w or f"{kind} {item.id} " in w]
        return list(dict.fromkeys([*item.review_flags, *q, *warned]))

    return {
        "_how": "Walk the place with the map. Set approve (edges: approve for source -> target, approve_reverse for "
                "the way back) on what you checked, fix name/kind/description/instruction, hide what is wrong, "
                "set merge_into (a node id) on a node that is part of another one, keep on what the cleaner left out "
                "(`pruned` in map.json) but should stay. Then run "
                "`nav map review <map_id> --apply <this file>` (or PUT it to /api/v1/maps/<map_id>/review). Only "
                "verified edges are used to route real users.",
        "nodes": {n.id: {"_": f"{n.name} ({n.kind}, {n.status})", "_check": flags(n, "node"),
                         **review.nodes.get(n.id, NodeReview()).model_dump()} for n in pmap.nodes},
        "edges": {e.id: {"_": f"{names.get(e.source)} -> {names.get(e.target)} ({e.status}, back: {e.reverse_status})",
                         "_instruction": e.instruction, "_check": flags(e, "edge"),
                         **review.edges.get(e.id, EdgeReview()).model_dump()} for e in pmap.edges},
        "relations": {r.id: {"_": f"at {n.name}: {r.subject} {r.direction} ({r.status})",
                             **review.relations.get(r.id, RelationReview()).model_dump()}
                      for n in pmap.nodes for r in n.relations},
        "_distances": "Metres measured on site between two node ids, e.g. "
                      '{"between": ["n0", "n2"], "meters": 7, "sigma_m": 0.3, "note": "tape"}. They set the scale '
                      "of every distance in the map (`nav map measure` adds one).",
        "distances": [d.model_dump() for d in review.distances],
        "_seen": "Names of the nodes this review refers to when it was saved: a rebuild that moves one of those places "
                 "to another id holds the entries that depend on it. Saving fills in missing names.",
        "seen": dict(review.seen),
    }


def from_template(data: dict) -> Review:
    """A Review from an edited template (keys starting with "_" are explanations, not review)."""
    def entries(kind: str, model) -> dict:
        out = {k: model(**{f: x for f, x in v.items() if not f.startswith("_")}) for k, v in data.get(kind, {}).items()}
        return {k: v for k, v in out.items() if v != model()}  # untouched items are not review

    distances = [MeasuredDistance(**{f: x for f, x in d.items() if not f.startswith("_")})
                 for d in data.get("distances", [])]
    return Review(nodes=entries("nodes", NodeReview), edges=entries("edges", EdgeReview),
                  relations=entries("relations", RelationReview), distances=distances,
                  seen={str(k): str(v) for k, v in data.get("seen", {}).items()})


def dumps(data: dict) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False, default=str)
