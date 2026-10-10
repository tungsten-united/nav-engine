"""The cleaner (nav/mapping/clean.py): the published map keeps only what the walks back up."""

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nav.map_api.app import create_app
from nav.mapping.build import published_map
from nav.mapping.clean import CleanParams, clean_map
from nav.mapping.review import EdgeReview, NodeReview, Review, save_review
from nav.navigation.localize import RefIndex
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


def test_a_weak_spot_on_the_way_stays_a_stop():
    # n2, seen by one walk, between n1 and n3 (e2 written the other way round): each hop is published as written
    m = pmap([node("n1"), node("n2", walks=1), node("n3")],
             [edge("e1", "n1", "n2", 2, 1, d=3.0), edge("e2", "n3", "n2", 1, 2, d=4.0)])
    c = clean_map(m)
    assert ids(c) == (["n1", "n2", "n3"], ["e1", "e2"])
    assert [h.instruction for h in plan_route(c, "n1", "n3", "observed")] == ["Walk to n2.", "Walk back to n3."]


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
    # n3 and n4 hang on edges no walk went along
    m = pmap([node("n1"), node("n2"), node("n3", walks=1), node("n4", walks=1)],
             [edge("e1", "n1", "n2", 2, 2), edge("e2", "n2", "n3", 0, 0, walked=False),
              edge("e4", "n1", "n4", 0, 0, walked=False)])
    assert ids(clean_map(m)) == (["n1", "n2"], ["e1"])
    review = Review(nodes={"n3": NodeReview(keep=True)},
                    edges={"e2": EdgeReview(keep=True), "e4": EdgeReview(keep=True)})
    assert ids(clean_map(m, review)) == (["n1", "n2", "n3", "n4"], ["e1", "e2", "e4"])
    kept = clean_map(m, Review(nodes={"n3": NodeReview(keep=True)}))  # kept without an edge: stays, with a warning
    assert "n3" in ids(kept)[0] and any("n3" in w for w in kept.warnings)
    m.edges[2].status = "verified"
    assert ids(clean_map(m))[1] == ["e1", "e4"]
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


def random_map(seed: int) -> PlaceMap:
    """A place of 3-9 spots and random edges: walked or not, each way 0-2 times, any confidence."""
    rng = random.Random(seed)
    n = rng.randint(3, 9)
    nodes = [node(f"n{i}", walks=rng.randint(1, 3), kind=rng.choice(["open_area"] * 4 + ["entrance", "room"]))
             for i in range(n)]
    edges = []
    for k in range(rng.randint(n - 1, 2 * n)):
        a, b = rng.sample(range(n), 2)
        walked = rng.random() < 0.8
        fwd, back = (rng.randint(1, 2), rng.randint(0, 2)) if walked else (0, 0)
        edges.append(edge(f"e{k}", f"n{a}", f"n{b}", fwd, back, d=round(rng.uniform(0.5, 8.0), 1), walked=walked,
                          confidence=rng.choice(["low", "medium", "high"])))
    return pmap(nodes, edges)


def geometry(m: PlaceMap, nodes: list[str]) -> list:
    """What localization measures between `nodes`: walked distances and neighbours (localize.py)."""
    index = RefIndex(m, np.zeros((0, 1), np.float32), [])
    return [({t: d for t, d in index.distances_from(s).items() if t in nodes}, index.neighbours(s) & set(nodes))
            for s in nodes]


@pytest.mark.parametrize("seed", range(200))
def test_cleaning_keeps_what_it_publishes_as_written_and_never_moves_spots(seed, tmp_path):
    m = random_map(seed)
    clean, raw = published_map(tmp_path, m, clean=True), published_map(tmp_path, m, clean=False)
    written = {e.id: e for e in m.edges}
    for e in clean.edges:  # whole edges, as written: an instruction is never joined, cut or moved
        assert e.model_dump(exclude={"review_flags"}) == written[e.id].model_dump(exclude={"review_flags"})
    kept = ids(clean)[0]
    assert {x for e in clean.edges for x in (e.source, e.target)} <= set(kept)
    assert geometry(clean, kept) == geometry(raw, kept)  # localization measures the same, cleaned or not


FIXTURE = Path(__file__).parent / "fixtures" / "itnig"


def test_itnig_as_published_on_2026_10_10(tmp_path):
    """Build 20261010T131002Z and its review (images, paths and tracks stripped). Cleaned with folding, the hop
    Drinks area -> Corridor end went past Right-side tables and its instruction joined two (263 characters); Orient
    spoke the first sentence, the user stopped at the tables, 4 m short, and heard "Please hold still" in a loop. And
    the walked-distance gate put the kitchen 5.5 m from the centre tables instead of 2.9 m."""
    shutil.copy(FIXTURE / "review.json", tmp_path)
    build = PlaceMap.model_validate_json((FIXTURE / "map.json").read_text())
    clean, raw = published_map(tmp_path, build, clean=True), published_map(tmp_path, build, clean=False)
    assert sorted(clean.pruned) == ["e11", "e12", "e13", "e14"]
    hops = plan_route(clean, "n2", "n6", "observed")
    assert [h.target for h in hops] == ["n5", "n6"]  # Right-side tables is a stop: each hop said whole
    assert hops[0].instruction == ("Bear right slightly and keep down the narrow aisle for about 3 metres until a "
                                   "group of tables is on your right.")
    assert not [w for w in clean.warnings if "characters" in w]
    kept = ids(clean)[0]
    assert geometry(clean, kept) == geometry(raw, kept)
    kitchen = RefIndex(clean, np.zeros((0, 1), np.float32), [])
    assert kitchen.distances_from("n7")["n10"] == 2.9 and "n10" in kitchen.neighbours("n7")


def test_instructions_orient_would_cut_are_flagged_until_the_review_shortens_them(tmp_path):
    m = pmap([node("n1"), node("n2")], [edge("e1", "n1", "n2", 2, 2)])
    m.edges[0].instruction = "Walk ahead. " * 21  # 252 characters
    flagged = published_map(tmp_path, m)
    assert flagged.edges[0].review_flags == ["instruction is 252 characters: Orient speaks up to 240, then stops"]
    assert flagged.warnings == ["edge e1: instruction is 252 characters: Orient speaks up to 240, then stops"]
    save_review(tmp_path, Review(edges={"e1": EdgeReview(instruction="Walk 5 metres ahead to the cooler.")}))
    fixed = published_map(tmp_path, m)
    assert fixed.edges[0].instruction == "Walk 5 metres ahead to the cooler." and not fixed.warnings


def test_glories_basement_walked_once_routes_back_only_over_any(tmp_path):
    """Glories Basement as published on 2026-10-10 (build 20261010T180958Z: one walk, entrance escalator to the food
    court, no review). Walking back with the app the user was placed right five times, but nav-api found no route
    over `observed`: 76 requests, none found. The orchestrator now routes over `any`, which on a published map means
    walked, either way: the way back has an instruction for that direction on every hop, within what Orient speaks,
    and the way in is the same route as before."""
    from nav.schemas.map import MAX_INSTRUCTION_CHARS

    build = PlaceMap.model_validate_json((Path(__file__).parent / "fixtures" / "glories-basement" / "map.json").read_text())
    m = published_map(tmp_path, build, clean=True)
    assert m.pruned == {} and len(m.nodes) == 16
    assert plan_route(m, "n13", "n1", "observed") is None
    back = plan_route(m, "n13", "n1", "any")
    assert [h.target for h in back] == ["n12", "n11", "n10", "n9", "n8", "n7", "n6", "n5", "n4", "n3", "n2", "n1"]
    assert all(not h.forward and h.status == "inferred" for h in back)
    assert all(h.instruction and len(h.instruction) <= MAX_INSTRUCTION_CHARS for h in back)
    way_in = [h.edge for h in plan_route(m, "n1", "n16", "observed")]
    assert [h.edge for h in plan_route(m, "n1", "n16", "any")] == way_in and len(way_in) == 15
