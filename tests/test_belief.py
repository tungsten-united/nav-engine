"""The belief over the map's nodes (nav/navigation/belief.py) and the loop that follows a route on it, on the Itnig
map of 2026-10-10 (tests/fixtures/itnig): route Main entrance n1 -> n2 -> n4 -> n5 -> n6 -> Kitchen n7."""

from pathlib import Path

import numpy as np

from nav.navigation.belief import BeliefLoop, BeliefParams, NodeBelief
from nav.navigation.route import plan_route
from nav.schemas.map import PlaceMap
from nav.schemas.navigation import Localization, NodeMatch, Thresholds

PMAP = PlaceMap.model_validate_json((Path(__file__).parent / "fixtures" / "itnig" / "map.json").read_text())
TH = Thresholds(confirm_score=0.45, confirm_margin=0.05, lost_score=0.35)
P = BeliefParams(temperature=0.03, gate=0.2, jump=0.01, walk_scale=1.0, walk_m_per_call=0.0, reach=0.7, leave=0.8)


def loc(scores: dict[str, float], implausible: tuple = ()) -> Localization:
    """A result where the nodes not named score 0.2."""
    allscores = {n.id: 0.2 for n in PMAP.nodes} | scores
    ranked = sorted(allscores.items(), key=lambda x: -x[1])
    cands = [NodeMatch(node=n, name=n, score=s, refs=[], plausible=False if n in implausible else None)
             for n, s in ranked]
    return Localization(map_id="itnig", model="x", frames=1, status="uncertain", reason="", best=ranked[0][0],
                        margin=ranked[0][1] - ranked[1][1], candidates=cands, thresholds=TH, took_ms={})


def path_from(node: str, t_s: float):
    return plan_route(PMAP, node, "n7", "observed")


def test_a_small_lead_that_keeps_up_adds_up_and_one_result_is_not_enough():
    b = NodeBelief(PMAP, P)
    b.update(loc({"n2": 0.33, "n3": 0.31}))
    assert b.best()[0] == "n2" and b.best()[1] < 0.7
    for _ in range(4):
        b.predict(0.0)
        b.update(loc({"n2": 0.33, "n3": 0.31}))
    assert b.best()[0] == "n2" and b.best()[1] > 0.7


def test_walking_moves_belief_to_linked_nodes_the_nearer_sooner_and_standing_keeps_it():
    b = NodeBelief(PMAP, BeliefParams(jump=0.0))
    b.p = np.eye(len(b.nodes))[b.ix["n2"]]
    b.predict(0.0)
    assert b.prob("n2") == 1.0
    b.predict(1.0)  # n2 links n1 (1.4 m), n4 (2.0 m), n3 (3.5 m), n5 (3.0 m)
    assert b.prob("n1") > b.prob("n4") > b.prob("n5") > b.prob("n3") > 0
    assert b.prob("n7") == 0.0  # not linked to n2
    assert abs(b.p.sum() - 1) < 1e-9


def test_a_node_nav_api_marks_farther_than_walked_counts_less():
    a, b = NodeBelief(PMAP, P), NodeBelief(PMAP, P)
    a.update(loc({"n6": 0.4}))
    b.update(loc({"n6": 0.4}, implausible=("n6",)))
    assert b.prob("n6") < a.prob("n6")


def test_locate_then_reach_a_node_further_on_the_camera_skipped_then_arrive():
    lp = BeliefLoop("n7", path_from, PMAP, P)
    assert lp.request()["top_k"] == len(PMAP.nodes)
    events = [lp.on_localization(loc({"n1": 0.35}), k, k) for k in range(4)]
    located = [e for e in events if e]
    assert [(e.kind, e.node) for e in located] == [("located", "n1")]
    assert [h.target for h in lp.path] == ["n2", "n4", "n5", "n6", "n7"]
    # the camera never shows n2, n4: n5 leading is enough, and the route goes on from it
    ev = next(e for k in range(4, 12) if (e := lp.on_localization(loc({"n5": 0.35}), 4 + 3 * k, k)))
    assert (ev.kind, ev.node) == ("reached", "n5")
    assert lp.request()["previous"] == "n5" and lp.request()["expected"] == "n6"
    ev = next(e for k in range(12, 30) if (e := lp.on_localization(loc({"n7": 0.35}), 40 + 3 * k, k)))
    assert (ev.kind, ev.node) == ("reached", "n6") or ev.kind == "arrived"
    while not lp.arrived:
        lp.on_localization(loc({"n7": 0.35}), 200, 99)
    assert lp.on_localization(loc({"n7": 0.35}), 200, 100) is None


def test_a_node_off_the_route_that_stays_ahead_drops_the_route():
    lp = BeliefLoop("n7", path_from, PMAP, P)
    for k in range(4):
        lp.on_localization(loc({"n1": 0.35}), k, k)
    assert lp.path is not None
    ev = next(e for k in range(4, 30) if (e := lp.on_localization(loc({"n11": 0.35}), k, k)))
    assert (ev.kind, ev.node) == ("lost", "n1") and lp.path is None


def test_without_steps_the_belief_still_moves_and_with_steps_standing_still_does_not():
    moving = BeliefLoop("n7", path_from, PMAP, BeliefParams(walk_m_per_call=1.0))
    still = BeliefLoop("n7", path_from, PMAP, BeliefParams(walk_m_per_call=1.0))
    for lp, steps in ((moving, None), (still, 5)):
        lp.belief.p = np.eye(len(lp.belief.nodes))[lp.belief.ix["n2"]]
        lp.on_localization(loc({}), steps, 0)
        lp.on_localization(loc({}), steps, 1)
    assert moving.belief.prob("n1") > 0.1
    assert still.belief.prob("n1") < 0.05  # only the jump


def test_one_candidate_still_tells_through_the_margin():
    """nav-api with top_k=1 (or a fake) lists only the leader: the others score at most the leader minus the margin."""
    b = NodeBelief(PMAP, P)
    one = loc({"n2": 0.4})
    one.candidates, one.margin = one.candidates[:1], 0.1
    b.update(one)
    assert b.best()[0] == "n2" and b.prob("n2") > 3 * b.prob("n3")
