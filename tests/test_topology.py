"""Where things are and how they connect, by code (nav/mapping/topology.py), and a map from that layout and the two
LLM passes (nav/mapping/assemble.py)."""

import json

import numpy as np
import pytest

from nav.mapping import topology
from nav.mapping.assemble import assemble, finish, layout_for, routes_input
from nav.schemas.map import Candidate, CandidateSet, Motion, PlacesDraft, Point, RoutesDraft, View
from test_navigation import BUILD


def path(points, heading=None, dt=500.0, rec="r1"):
    xy = np.array(points, float)
    h = np.full(len(xy), np.nan) if heading is None else np.array(heading, float)
    return topology.WalkPath(rec, np.arange(len(xy)) * dt, xy, h, np.ones(len(xy), bool))


def test_spots_fold_by_position_with_complete_linkage():
    pts = [(0, 0), (1.5, 0), (3.0, 0), (20, 0), None]
    assert topology.fold_by_position(pts) == [[0, 1], [2], [3], [4]]  # 0-1-2 is a chain, not one spot


def test_a_node_keeps_the_spots_within_reach_of_each_other():
    pos = {"c1": (0, 0), "c2": (2, 0), "c3": (3, 1), "c4": (30, 0)}
    keep, out = topology.node_spots(["c1", "c2", "c3", "c4", "c5"], pos)
    assert keep == ["c1", "c2", "c3", "c5"] and out == ["c4"]  # c5 has no position: kept
    assert topology.node_position(keep, pos) == pytest.approx((5 / 3, 1 / 3))


def test_visits_and_passes_follow_the_walk_from_node_to_node():
    xs = [(x / 2, 0.0) for x in range(41)]  # 0 to 20 m east, 0.5 m a sample, 2 samples a second
    p = path(xs, heading=[90.0] * 41)
    nodes = {"a": (0.0, 0.0), "b": (10.0, 0.0), "c": (20.0, 0.0)}
    radius = {n: 2.5 for n in nodes}
    vs = topology.visits(p, nodes, radius)
    assert [v.node for v in vs] == ["a", "b", "c"]
    ps = topology.passes({"r1": p}, nodes, radius)
    assert [(q.a, q.b) for q in ps] == [("a", "b"), ("b", "c")]
    assert ps[0].length_m(p) == pytest.approx(5.0)  # from leaving a's radius to entering b's


def test_a_glance_at_a_node_on_the_way_is_not_a_visit():
    xs = [(x / 2, 3.0 if 18 <= x <= 22 else 6.0) for x in range(41)]  # passes 3 m from b for 1 s at most
    p = path(xs)
    nodes = {"b": (10.0, 0.0)}
    assert topology.visits(p, nodes, {"b": 3.0}) == []


def test_steps_turn_where_the_walk_bends():
    l_shape = [(x, 0.0) for x in range(11)] + [(10.0, -y) for y in range(1, 11)]  # east 10 m, then south 10 m
    steps = topology.skeleton(np.array(l_shape, float), departure_turn=0.0)
    assert [(s.action, s.distance_m) for s in steps] == [("straight", 10.0), ("turn_right", 10.0)]
    back = topology.skeleton(np.array(l_shape[::-1], float), departure_turn=180.0)
    assert [(s.action, s.distance_m) for s in back] == [("turn_around", 10.0), ("turn_left", 10.0)]
    wiggle = [(x, 0.3 * (x % 2)) for x in range(11)]  # small sideways steps are no bends
    assert len(topology.skeleton(np.array(wiggle, float), None)) == 1


def test_turn_actions():
    assert [topology.turn_action(d) for d in (None, 10, -30, 80, -100, 170)] == \
        ["straight", "straight", "bear_left", "turn_right", "turn_left", "turn_around"]


def test_relations_say_where_the_other_nodes_are_from_the_way_walks_arrived():
    positions = {"n1": (0.0, 0.0), "n2": (10.0, 0.0), "n3": (0.0, 10.0)}
    names = {"n1": "Door", "n2": "Kitchen", "n3": "Stage"}
    rel = topology.node_relations("n1", names, positions, arrival=0.0, view=("c1", "r1", 0.0, "img", 0.0))
    by = {r.subject: r for r in rel}
    assert by["Kitchen"].heading_deg == 90 and by["Kitchen"].direction == "right"
    assert by["Stage"].heading_deg == 0 and by["Stage"].direction == "ahead" and by["Stage"].status == "observed"


# --- assemble: code's layout and the LLM's words ---------------------------------------------------------------
def l_build(tmp_path):
    """A build without a 3D model (dead reckoning only): one walk east 10 m then south 10 m, candidates at its
    corners. -> (build_dir, cset, places)."""
    def view(t, x, y, h):
        return View(rec_id="r1", t_ms=t * 1000, sharpness=1,
                    motion=Motion(walked_m=t, heading_deg=h, x_m=x, y_m=y, pitch_deg=0, moving=True))
    build = tmp_path / "maps" / "t" / "builds" / BUILD.build_id
    build.mkdir(parents=True)
    pts = [[t * 500, min(t / 2, 10.0), -max(0.0, t / 2 - 10), 90.0 if t / 2 < 10 else 180.0] for t in range(41)]
    (build / "tracks.json").write_text(json.dumps({"r1": pts}))
    cands = [Candidate(id="c01", image="frames/c01.jpg", reason=[], views=[view(0, 0, 0, 90)],
                       position=Point(x_m=0, y_m=0)),
             Candidate(id="c02", image="frames/c02.jpg", reason=[], views=[view(10, 10, 0, 90)],
                       position=Point(x_m=10, y_m=0)),
             Candidate(id="c03", image="frames/c03.jpg", reason=[], views=[view(20, 10, -10, 180)],
                       position=Point(x_m=10, y_m=-10)),
             Candidate(id="c04", image="frames/c04.jpg", reason=[], views=[view(20, 10, -10, 180)],
                       position=Point(x_m=40, y_m=-10))]
    from test_navigation import lshape

    rec = lshape()[1].recordings[0]
    cset = CandidateSet(place="T", map_id="t", recordings=[rec], candidates=cands, params={}, embed_model="m")
    node = lambda i, cs, rel=(): {"id": f"n{i}", "name": f"N{i}", "kind": "room", "description": "",  # noqa: E731
                                  "landmarks": [f"cue {i}"], "hazards": [], "visual_cues": [], "floor": None,
                                  "candidates": cs, "relations": list(rel), "confidence": "high"}
    places = PlacesDraft.model_validate({
        "place": "T", "language": "en", "summary": "s", "notes": ["n"],
        "nodes": [node(1, ["c01"]), node(2, ["c02"], [{"subject": "Counter", "candidate": "c02", "side": "in_photo",
                                                        "x": 0.5, "evidence": "observed"},
                                                       {"subject": "N1", "candidate": "c02", "side": "behind",
                                                        "x": None, "evidence": "inferred"}]),
                  node(3, ["c03", "c04"])]})
    return build, cset, places


def test_the_layout_places_nodes_and_finds_the_walked_edges(tmp_path):
    build, cset, places = l_build(tmp_path)
    lay = layout_for(build, cset, places)
    assert [n.candidates for n in lay.nodes] == [["c01"], ["c02"], ["c03"]]  # c04 is 30 m away: left out
    assert any("c04" in w for w in lay.warnings)
    assert [(e.source, e.target) for e in lay.edges] == [("n1", "n2"), ("n2", "n3")]
    e1, e2 = lay.edges
    assert e1.walked_forward == 1 and e1.walked_back == 0 and e1.distance_m == pytest.approx(10.0)
    assert [s.action for s in e2.steps][0] == "turn_right" and e2.turn_deg == 90
    r = routes_input(lay, cset, None)
    assert [e["id"] for e in r["edges"]] == ["e1", "e2"] and r["edges"][1]["source_name"] == "N2"
    assert r["edges"][1]["steps"][0]["action"] == "turn_right" and r["edges"][0]["previous"] is None


def test_the_routes_pass_fills_the_words_but_not_the_turns(tmp_path):
    build, cset, places = l_build(tmp_path)
    lay = layout_for(build, cset, places)
    e2_steps = lay.edges[1].steps
    routes = RoutesDraft.model_validate({"notes": ["r"], "edges": [
        {"id": "e1", "kind": "door", "instruction": "Walk ahead.", "reverse_instruction": "Back.", "landmarks": ["wall"],
         "hazards": [], "visual_cues": [], "confidence": "high",
         "steps": [{"action": "go_through", "along": "wall on your left", "until": "the floor turns to carpet"}]},
        {"id": "e2", "kind": "walk", "instruction": "Turn left.", "reverse_instruction": "x", "landmarks": [],
         "hazards": [], "visual_cues": [], "confidence": "medium",
         "steps": [{"action": "turn_left", "along": None, "until": "the bins"}] * len(e2_steps)},
        {"id": "e9", "kind": "walk", "instruction": "?", "reverse_instruction": "?", "landmarks": [], "hazards": [],
         "visual_cues": [], "confidence": "low", "steps": []}]})
    warnings = []
    draft = assemble(places, routes, lay, warnings)
    e1, e2 = draft.edges
    assert e1.kind == "door" and e1.steps[0].action == "go_through" and e1.steps[0].until.startswith("the floor")
    assert e2.steps[0].action == "turn_right" and e2.steps[0].until == "the bins"  # the walk turned right
    assert any("e2: step 1 is turn_right" in w for w in warnings) and any("e9" in w for w in warnings)
    assert draft.notes == ["n", "r"] and all(e.basis == "walked" for e in draft.edges)
    no_words = assemble(places, None, lay, [])
    assert no_words.edges[0].instruction == "Go straight and walk about 10 metres, to N2."

    m = finish(draft, cset, lay, BUILD, "builds/b", json.loads((build / "tracks.json").read_text()))
    n1, n2, n3 = m.nodes
    assert (n2.position.x_m, n2.position.y_m) == (10.0, 0.0) and n2.evidence[0].image == "builds/b/frames/c02.jpg"
    subjects = [r.subject for r in n2.relations]
    assert subjects.count("N1") == 1 and "Counter" in subjects  # code's N1, not the LLM's; the counter kept
    assert next(r for r in n2.relations if r.subject == "N1").direction == "behind"  # arrived facing east
    me1 = m.edges[0]
    assert me1.walked and me1.distance_m == pytest.approx(10.0) and me1.status == "observed"
    assert me1.reverse_status == "inferred" and "the way back was not walked" in me1.review_flags
    assert me1.reverse_steps and me1.departure_heading_deg == 90 and m.tracks[0].rec_id == "r1"
