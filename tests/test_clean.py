"""The cleaner (nav/mapping/clean.py): the published map keeps only what the walks back up."""

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from nav.map_api.app import create_app
from nav.mapping.clean import CleanParams, clean_map
from nav.mapping.review import EdgeReview, NodeReview, Review
from nav.mapping.store import MapStore
from nav.navigation.route import plan_route
from nav.recorder.store import RecordingStore
from nav.schemas.map import (BuildInfo, Evidence, LlmRun, PlaceEdge, PlaceMap, PlaceNode, Point, ReferenceImage,
                             Step)

BUILD = BuildInfo(build_id="20261010T000000Z", built_at_utc=datetime(2026, 10, 10, tzinfo=timezone.utc),
                  pipeline_version="t", params={}, embed_model="m", transcribe_model=None,
                  llm=LlmRun(backend="claude_code", model="x"), candidates=0)


def node(i: str, walks: int = 2, kind: str = "open_area", x: float = 0.0) -> PlaceNode:
    ev = [Evidence(candidate="c", rec_id=f"r{k}", t_ms=0.0, image="x.jpg") for k in range(walks)]
    return PlaceNode(id=i, name=i.upper(), kind=kind, description="", landmarks=[], hazards=[], visual_cues=[],
                     floor=None, candidates=[], relations=[], confidence="high", evidence=ev,
                     position=Point(x_m=x, y_m=0.0))


def edge(i: str, a: str, b: str, fwd: int = 1, back: int = 1, d: float = 5.0, walked: bool = True,
         confidence: str = "high") -> PlaceEdge:
    return PlaceEdge(id=i, source=a, target=b, kind="walk", bidirectional=True, basis="walked",
                     steps=[Step(action="straight", distance_m=d, along=None, until=f"at {b}")],
                     reverse_steps=[Step(action="turn_around", distance_m=d, along=None, until=f"at {a}")],
                     instruction=f"Walk to {b}.", reverse_instruction=f"Walk back to {a}.", landmarks=[], hazards=[],
                     visual_cues=[], candidates=[], confidence=confidence, distance_m=d, walked=walked,
                     path=[Point(x_m=0, y_m=0), Point(x_m=d, y_m=0)],
                     status="observed" if fwd else "inferred", reverse_status="observed" if back else "inferred",
                     walked_forward=fwd, walked_back=back)


def pmap(nodes, edges, refs=()) -> PlaceMap:
    return PlaceMap(place="T", language="en", summary="", nodes=nodes, edges=edges, notes=[], map_id="t",
                    recordings=["r0", "r1"], build=BUILD, references=list(refs))


def ids(m: PlaceMap) -> tuple[list[str], list[str]]:
    return [n.id for n in m.nodes], [e.id for e in m.edges]


def test_inferred_and_duplicate_edges_go():
    m = pmap([node("n1"), node("n2"), node("n3")],
             [edge("e1", "n1", "n2"), edge("e2", "n2", "n3"), edge("e3", "n1", "n3", 0, 0, walked=False),
              edge("e4", "n2", "n1", fwd=2, back=1)])
    c = clean_map(m)
    assert ids(c) == (["n1", "n2", "n3"], ["e2", "e4"])
    assert c.pruned["e3"].startswith("inferred") and "e4" in c.pruned["e1"]
    assert (c.clean.nodes_before, c.clean.edges_before, c.clean.edges) == (3, 4, 2)
    assert len(m.edges) == 4  # the input is untouched


def test_a_duplicate_walked_the_other_way_stays():
    # after a merge, e1 (n1 -> n2 walked twice) and e2 (n2 -> n1 walked once) join the same spots, each one way
    m = pmap([node("n1"), node("n2")], [edge("e1", "n1", "n2", 2, 0), edge("e2", "n2", "n1", 1, 0)])
    c = clean_map(m)
    assert ids(c)[1] == ["e1", "e2"] and plan_route(c, "n2", "n1", "observed")
    # walked back as well, the stronger one covers both ways: the weaker goes
    m.edges[0].reverse_status, m.edges[0].walked_back = "observed", 1
    assert ids(clean_map(m))[1] == ["e1"]


def test_a_weak_spot_on_the_way_is_folded_into_one_edge():
    refs = [ReferenceImage(id="r1", node="n1", edge="e1", toward="n2", role="outgoing", image="x", rec_id="r0",
                           t_ms=0, heading_deg=None),
            ReferenceImage(id="r2", node="n2", role="at", image="x", rec_id="r0", t_ms=0, heading_deg=None)]
    # e2 was written the other way round (n3 -> n2): it is turned before chaining
    m = pmap([node("n1"), node("n2", walks=1), node("n3")],
             [edge("e1", "n1", "n2", 2, 1, d=3.0), edge("e2", "n3", "n2", 1, 2, d=4.0)], refs)
    m.edges[0].reference_views = ["r1"]
    c = clean_map(m)
    assert ids(c) == (["n1", "n3"], ["e1"])
    e = c.edges[0]
    assert (e.source, e.target, e.via, e.distance_m) == ("n1", "n3", ["n2"], 7.0)
    assert [s.until for s in e.steps] == ["at n2", "at n3"]  # e1's steps, then e2's way back (n2 -> n3)
    assert e.instruction == "Walk to n2. Then, walk back to n3."
    assert (e.walked_forward, e.walked_back) == (2, 1)
    assert [r.id for r in c.references] == ["r1"] and c.references[0].toward == "n3"
    assert c.pruned["n2"].startswith("seen by 1 walk") and "e1" in c.pruned["e2"]


def test_weak_edges_stay_only_to_keep_the_map_walkable():
    # core: n1 - n2 (2 walks). n3 hangs on two single walks, n2 -> n3 (e2) and n3 -> n1 (e3), and so does a
    # protected destination seen once (n4): each is reached one way and left the other.
    m = pmap([node("n1"), node("n2"), node("n3"), node("n4", walks=1, kind="destination")],
             [edge("e1", "n1", "n2", 1, 1), edge("e2", "n2", "n3", 1, 0, d=3.0), edge("e3", "n3", "n1", 1, 0),
              edge("e5", "n2", "n4", 1, 0, d=2.0), edge("e6", "n4", "n1", 1, 0, d=2.0)])
    c = clean_map(m)
    assert ids(c)[0] == ["n1", "n2", "n3", "n4"]  # the destination stays
    for start, goal in [("n1", "n3"), ("n3", "n1"), ("n1", "n4"), ("n4", "n1")]:
        assert plan_route(c, start, goal, "observed"), (start, goal)
    assert any("walked 1 time" in f for e in c.edges for f in e.review_flags)

    # without connecting, only the core survives and the weak spots fall away
    bare = clean_map(m, params=CleanParams(connect=False))
    assert ids(bare) == (["n1", "n2", "n4"], ["e1"]) and any("n4" in w for w in bare.warnings)


def test_a_core_edge_longer_than_the_walked_way_around_goes():
    m = pmap([node("n1"), node("n2"), node("n3")],
             [edge("e1", "n1", "n2", 2, 2, d=2.0), edge("e2", "n2", "n3", 2, 2, d=2.0),
              edge("e3", "n1", "n3", 2, 2, d=6.0)])
    c = clean_map(m)
    assert ids(c)[1] == ["e1", "e2"] and "longer" not in c.pruned["e3"] and "no shorter" in c.pruned["e3"]
    # walked one way only, the way around can't replace it the other way: it stays
    m.edges[1].status = "inferred"
    assert "e3" in ids(clean_map(m))[1]


def test_review_keep_and_verified_items_are_never_pruned():
    m = pmap([node("n1"), node("n2"), node("n3", walks=1), node("n4", walks=1)],
             [edge("e1", "n1", "n2", 2, 2), edge("e2", "n2", "n3", 1, 0), edge("e3", "n3", "n1", 1, 0),
              edge("e4", "n1", "n4", 0, 0, walked=False)])
    assert "n3" in clean_map(m).pruned and "e4" in clean_map(m).pruned
    review = Review(nodes={"n3": NodeReview(keep=True)}, edges={"e4": EdgeReview(keep=True)})
    c = clean_map(m, review)
    assert ids(c) == (["n1", "n2", "n3", "n4"], ["e1", "e2", "e3", "e4"])  # n3 in by e2, out by e3
    m.edges[3].status = "verified"
    assert "e4" in ids(clean_map(m))[1]
    # a note alone is not a reason to keep
    assert "n3" in clean_map(m, Review(nodes={"n3": NodeReview(note="looked at it")})).pruned


def test_republish_and_the_unclean_view(tmp_path, monkeypatch):
    m = pmap([node("n1"), node("n2"), node("n3")],
             [edge("e1", "n1", "n2", 2, 2), edge("e2", "n2", "n3", 2, 2), edge("e3", "n1", "n3", 0, 0, walked=False)])
    build = tmp_path / "maps" / "t" / "builds" / BUILD.build_id
    build.mkdir(parents=True)
    from nav.mapping.build import write_map

    published = write_map(build, m)
    assert ids(published)[1] == ["e1", "e2"] and published.pruned
    assert len(json.loads((build / "map.json").read_text())["edges"]) == 3  # the build keeps everything

    app = create_app(RecordingStore(tmp_path / "recordings"), MapStore(tmp_path / "maps"), api_token=None,
                     cache_dir=tmp_path / "cache", web_dir=None)
    with TestClient(app) as c:
        assert len(c.get("/api/v1/maps/t").json()["edges"]) == 2
        raw = c.get("/api/v1/maps/t?clean=false").json()
        assert len(raw["edges"]) == 3 and raw["clean"] is None
        from nav import config

        monkeypatch.setattr(config, "MAP_CLEAN", False)
        r = c.post("/api/v1/maps/t/republish").json()
        assert len(r["edges"]) == 3 and r["pruned"] == {}
        assert c.post("/api/v1/maps/nope/republish").status_code == 404
