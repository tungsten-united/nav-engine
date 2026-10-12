"""A map from code's layout and the LLM's words (pipeline metric-1; build.py runs it).

    layout       the places pass's nodes, kept to spots within reach of each other (topology.node_spots), placed at
                 their spots, and the edges the walks went along between them, with their geometry
    routes.json  what the routes pass writes about: the nodes, and each edge with its code steps (actions, metres),
                 the frames along it, and the previous map's instruction for the same connection
    assemble     the MapDraft: the places pass's nodes, and code's edges with the routes pass's words on them (a step's
                 action only changes from straight to a door, step, stairs or lift; turns and metres stay code's)
    finish       the PlaceMap: evidence, positions, each edge's walked path, distance, turn, headings and status from
                 the passes that walked it, node-to-node relations as bearings, and the walks' paths for drawing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nav.mapping import topology
from nav.schemas.map import (
    BuildInfo,
    CandidateSet,
    Evidence,
    MapDraft,
    MapEdge,
    MapNode,
    PlaceEdge,
    PlaceMap,
    PlaceNode,
    Point,
    PlacesDraft,
    RoutesDraft,
    Step,
    Track,
)

log = logging.getLogger(__name__)

EVENT_ACTIONS = {"go_through", "step_up", "step_down", "stairs_up", "stairs_down", "lift"}


@dataclass
class Layout:
    nodes: list[MapNode]  # the places pass's, with the candidates code kept
    positions: dict[str, tuple[float, float]]
    paths: dict[str, topology.WalkPath]
    passes: list[topology.Pass]
    edges: list[topology.EdgeGeometry]
    candidate_positions: dict[str, tuple[float, float]]
    warnings: list[str] = field(default_factory=list)
    geometry: dict | None = None  # the 3D model's info, when there is one

    def passes_between(self, a: str, b: str) -> list[topology.Pass]:
        return [p for p in self.passes if {p.a, p.b} == {a, b}]


def walk_paths(build_dir: Path, cset: CandidateSet) -> tuple[dict[str, topology.WalkPath], dict | None]:
    """The walks' paths in the map frame: the build's 3D model, else the dead-reckoned tracks aligned through the
    candidates the walks share."""
    from nav.mapping import geometry
    from nav.mapping.build import align_recordings

    geo = geometry.load(build_dir)
    if geo is not None:
        return topology.paths_from_geometry(geo), geo.info
    tracks = json.loads((build_dir / "tracks.json").read_text()) if (build_dir / "tracks.json").is_file() else {}
    offsets, _ = align_recordings({c.id: c.views for c in cset.candidates})
    return topology.paths_from_tracks(tracks, offsets), None


def layout_for(build_dir: Path, cset: CandidateSet, places: PlacesDraft) -> Layout:
    warnings: list[str] = []
    paths, info = walk_paths(build_dir, cset)
    cpos = {c.id: (c.position.x_m, c.position.y_m) for c in cset.candidates if c.position is not None}
    known = {c.id for c in cset.candidates}
    nodes, positions, radius = [], {}, {}
    for n in places.nodes:
        if any(x.id == n.id for x in nodes):
            warnings.append(f"duplicate node id {n.id} dropped")
            continue
        unknown = [c for c in n.candidates if c not in known]
        if unknown:
            warnings.append(f"node {n.id} cites unknown candidates {', '.join(unknown)}")
        cids = [c for c in n.candidates if c in known]
        keep, out = topology.node_spots(cids, cpos)
        if out:
            warnings.append(f"node {n.id} ({n.name}): candidates {', '.join(out)} are more than "
                            f"{topology.NODE_SPAN_M:.0f} m from the rest: left out")
        if not keep:
            warnings.append(f"node {n.id} ({n.name}) has no candidates: dropped")
            continue
        nodes.append(n.model_copy(update={"candidates": keep}))
        if (pos := topology.node_position(keep, cpos)) is not None:
            positions[n.id] = pos
            radius[n.id] = topology.node_radius(keep, cpos)
        else:
            warnings.append(f"node {n.id} ({n.name}) has no position: no edges")
    if unused := [c for c in known if not any(c in n.candidates for n in nodes)]:
        warnings.append(f"candidates not used by any node: {', '.join(sorted(unused))}")
    ps = topology.passes(paths, positions, radius)
    edges = topology.edges_from_passes(ps, paths, positions, cset.candidates)
    edges.sort(key=lambda e: (min(p.rec_id for p in e.passes), min(p.t_leave for p in e.passes)))
    return Layout(nodes, positions, paths, ps, edges, cpos, warnings, info)


def edge_ids(layout: Layout) -> list[str]:
    return [f"e{i + 1}" for i in range(len(layout.edges))]


def routes_input(layout: Layout, cset: CandidateSet, prev: dict | None) -> dict:
    """routes.json: the nodes and code's edges, for the routes pass."""
    from nav.mapping.continuity import match_previous

    names = {n.id: n.name for n in layout.nodes}
    prev_instr: dict[tuple[str, str], str] = {}
    if prev:
        to_prev = match_previous(layout.nodes, cset, prev)
        from_prev = {v: k for k, v in to_prev.items()}
        for e in prev.get("edges", []):
            s, t = from_prev.get(e["source"]), from_prev.get(e["target"])
            if s and t:
                if e.get("instruction"):
                    prev_instr[(s, t)] = e["instruction"]
                if e.get("reverse_instruction"):
                    prev_instr.setdefault((t, s), e["reverse_instruction"])
    cands = {c.id: c for c in cset.candidates}
    edges = []
    for eid, g in zip(edge_ids(layout), layout.edges):
        along = []
        for cid in g.candidates:
            v = next((v for v in cands[cid].views if v.rec_id == g.best.rec_id), cands[cid].views[0])
            along.append({"id": cid, "rec_id": v.rec_id, "t_ms": v.t_ms, "marks": v.marks, "voice": v.voice})
        edges.append({"id": eid, "source": g.source, "target": g.target, "source_name": names[g.source],
                      "target_name": names[g.target], "distance_m": g.distance_m,
                      "steps": [{"action": s.action, "distance_m": s.distance_m} for s in g.steps],
                      "walked": {"forward": g.walked_forward, "back": g.walked_back}, "candidates": along,
                      "previous": prev_instr.get((g.source, g.target))})
    return {"place": cset.place,
            "nodes": [{"id": n.id, "name": n.name, "kind": n.kind, "description": n.description,
                       "landmarks": n.landmarks} for n in layout.nodes],
            "edges": edges}


def _template(steps: list[Step], target: str) -> str:
    """A plain instruction when the routes pass has none for an edge."""
    words = {"straight": "Go straight", "bear_left": "Bear left", "bear_right": "Bear right",
             "turn_left": "Turn left", "turn_right": "Turn right", "turn_around": "Turn around",
             "go_through": "Go through", "step_up": "Step up", "step_down": "Step down", "stairs_up": "Go up the stairs",
             "stairs_down": "Go down the stairs", "lift": "Take the lift"}
    parts = [f"{words.get(s.action, 'Go')}" + (f" and walk about {round(s.distance_m)} metres" if s.distance_m else "")
             for s in steps]
    return ". ".join(parts) + f", to {target}."


def assemble(places: PlacesDraft, routes: RoutesDraft | None, layout: Layout, warnings: list[str]) -> MapDraft:
    names = {n.id: n.name for n in layout.nodes}
    said = {r.id: r for r in (routes.edges if routes else [])}
    edges = []
    for eid, g in zip(edge_ids(layout), layout.edges):
        r = said.get(eid)
        steps = []
        for k, s in enumerate(g.steps):
            rs = r.steps[k] if r and k < len(r.steps) else None
            action = s.action
            if rs is not None and rs.action != s.action:
                if s.action == "straight" and rs.action in EVENT_ACTIONS:
                    action = rs.action
                else:
                    warnings.append(f"edge {eid}: step {k + 1} is {s.action} from the walks, the text said "
                                    f"{rs.action}: kept {s.action}")
            steps.append(Step(action=action, distance_m=s.distance_m, along=rs.along if rs else None,
                              until=rs.until if rs else None))
        if r is not None and len(r.steps) != len(g.steps):
            warnings.append(f"edge {eid}: {len(r.steps)} steps written for {len(g.steps)}")
        if r is None:
            warnings.append(f"edge {eid} ({names[g.source]} -> {names[g.target]}): no text from the routes pass, "
                            "a plain instruction instead")
        edges.append(MapEdge(
            id=eid, source=g.source, target=g.target, kind=r.kind if r else "walk", bidirectional=True,
            basis="walked", steps=steps,
            instruction=r.instruction if r else _template(steps, names[g.target]),
            reverse_instruction=r.reverse_instruction if r else None,
            landmarks=r.landmarks if r else [], hazards=r.hazards if r else [],
            visual_cues=r.visual_cues if r else [], candidates=g.candidates,
            confidence=r.confidence if r else "low"))
    unknown = [r.id for r in (routes.edges if routes else []) if r.id not in set(edge_ids(layout))]
    if unknown:
        warnings.append(f"routes pass wrote unknown edges {', '.join(unknown)}: ignored")
    return MapDraft(place=places.place, language=places.language, summary=places.summary, nodes=layout.nodes,
                    edges=edges, notes=places.notes + (routes.notes if routes else []))


def finish(draft: MapDraft, cset: CandidateSet, layout: Layout, build: BuildInfo, build_rel: str,
           tracks: dict[str, list] | None = None) -> PlaceMap:
    """The stored map (see the module doc). `draft` has stable ids already (continuity.stabilize); layout ids are
    the draft's before that, so nodes are matched back by their candidates."""
    from nav.mapping.build import (_relations, instruction_kind, map_frame, reverse_steps, visual_word_warnings)
    from nav.schemas.map import MapFrame

    warnings = visual_word_warnings(draft) + list(layout.warnings)
    cands = {c.id: c for c in cset.candidates}
    by_cands = {frozenset(n.candidates): n.id for n in layout.nodes}  # layout id by its candidates
    lid = {n.id: by_cands.get(frozenset(n.candidates)) for n in draft.nodes}  # stable id -> layout id
    positions = {n.id: layout.positions[lid[n.id]] for n in draft.nodes if lid[n.id] in layout.positions}
    names = {n.id: n.name for n in draft.nodes}

    nodes = []
    for n in draft.nodes:
        evidence = [Evidence(candidate=c, rec_id=v.rec_id, t_ms=v.t_ms, image=f"{build_rel}/{cands[c].image}")
                    for c in n.candidates for v in cands[c].views]
        pos = positions.get(n.id)
        nodes.append(PlaceNode(**{**n.model_dump(exclude={"relations"}), "candidates": n.candidates},
                               evidence=evidence, position=None if pos is None else Point(x_m=round(pos[0], 2),
                                                                                         y_m=round(pos[1], 2))))
    by_id = {n.id: n for n in nodes}
    to_stable = {v: k for k, v in lid.items() if v}
    passes = [topology.Pass(p.rec_id, to_stable[p.a], to_stable[p.b], p.i_leave, p.i_reach, p.t_leave, p.t_reach)
              for p in layout.passes if p.a in to_stable and p.b in to_stable]

    edges = []
    for e in draft.edges:
        ps = [p for p in passes if {p.a, p.b} == {e.source, e.target}]
        pe = PlaceEdge(**e.model_dump())
        if not ps or e.source not in positions or e.target not in positions:
            warnings.append(f"edge {e.id}: no walk between {e.source} and {e.target}: dropped")
            continue
        g = topology.edge_geometry(e.source, e.target, ps, layout.paths, positions, cset.candidates)
        pe.walked, pe.basis = True, "walked"
        pe.path, pe.distance_m, pe.measured_distance_m = g.path, g.distance_m, g.distance_m
        pe.walked_forward, pe.walked_back = g.walked_forward, g.walked_back
        pe.walked_in = g.best.rec_id
        pe.walked_at_ms = g.best.t_leave if g.best.a == e.source else g.best.t_reach
        pe.turn_deg = g.turn_deg
        pe.departure_heading_deg, pe.arrival_heading_deg = g.departure_deg, g.arrival_deg
        src = by_id[e.source]
        pe.reverse_steps = reverse_steps(pe.steps, arrival=src.landmarks[0] if src.landmarks else None)
        pe.status = "observed" if pe.walked_forward else "inferred"
        pe.reverse_status = "observed" if pe.walked_back else "inferred"
        pe.instruction_kind = instruction_kind(pe)
        if not pe.walked_forward:
            pe.review_flags.append("only walked from the target to the source")
        elif not pe.walked_back:
            pe.review_flags.append("the way back was not walked")
        edges.append(pe)

    arrivals = {n.id: topology.arrival_heading(n.id, passes, layout.paths) for n in nodes}
    fovs = {r.rec_id: r.fov_deg for r in cset.recordings if r.fov_deg}
    drafted = {n.id: n for n in draft.nodes}
    node_names = {nm.casefold() for nm in names.values()}
    for n in nodes:
        n.review_flags = ([f"{n.confidence} confidence"] if n.confidence != "high" else []) + \
                         ([] if n.landmarks else ["no arrival cue"])
        first = cands[n.candidates[0]]
        v = first.views[0]
        view = (first.id, v.rec_id, v.t_ms, f"{build_rel}/{first.image}", v.motion.heading_deg)
        rel = topology.node_relations(n.id, names, positions, arrivals[n.id], view)
        things = [r for r in drafted[n.id].relations if r.subject.casefold() not in node_names]
        n.relations = rel + _relations(n, things, cands, tracks or {}, fovs, build_rel, warnings)

    placed = []
    for rec, path in layout.paths.items():
        pts, last = [], -1e18
        for t, (x, y) in zip(path.t, path.xy):
            if np.isnan(x) or t - last < 1000:
                continue
            pts.append((round(float(t)), round(float(x), 2), round(float(y), 2)))
            last = t
        if pts:
            placed.append(Track(rec_id=rec, points=pts))
    frame = map_frame(cset.recordings) or MapFrame(heading_source="unknown", heading_zero_abs_deg=None,
                                                   step_length_m=None)
    if layout.geometry:
        frame.note = ("3D model of the walks' video (structure from motion), scaled by the walks' floor odometry "
                      "and turned to their compass headings; where it has no pose, dead reckoning")
    return PlaceMap(tracks=placed, **draft.model_dump(exclude={"nodes", "edges"}), map_id=cset.map_id,
                    recordings=[r.rec_id for r in cset.recordings], build=build, nodes=nodes, edges=edges,
                    warnings=warnings, frame=frame, geometry=layout.geometry)
