"""The model of the orchestrator's navigation loop (nav/navigation/replay.py, OrchestratorLoop): locate, one route,
then along it on votes over /localize results (the hop's target, or a node further on); lost, start over. The route is
n1 -> n2 -> n3 -> n4."""

from pathlib import Path

from nav.navigation.pdr import walk_route, walk_visits
from nav.navigation.replay import LoopParams, OrchestratorLoop, follow_vote, locate_vote
from nav.navigation.route import Hop
from nav.schemas.map import PlaceMap
from nav.schemas.navigation import Localization, NodeMatch, Thresholds

TH = Thresholds(confirm_score=0.45, confirm_margin=0.05, lost_score=0.35)
PATH = [Hop(f"e{i}", f"n{i}", f"n{i + 1}", True, 3.0, None, None) for i in (1, 2, 3)]


def loc(best: str | None, status: str = "uncertain", margin: float = 0.05, plausible: bool | None = None,
        score: float = 0.4) -> Localization:
    cands = [] if best is None else [NodeMatch(node=best, name=best, score=score, refs=[], plausible=plausible)]
    return Localization(map_id="m", model="x", frames=2, status=status, reason="", best=best, margin=margin,
                        candidates=cands, thresholds=TH, took_ms={})


def path_from(node: str, t_s: float):
    k = next((i for i, h in enumerate(PATH) if h.source == node), None)
    return None if k is None else PATH[k:]


def test_votes():
    assert locate_vote(loc("n2"), 0.02) == "n2"
    assert locate_vote(loc("n2", margin=0.01), 0.02) is None  # too close to the second
    assert locate_vote(loc("n2", status="lost"), 0.02) is None
    # locating, a lost result votes when it leads clearly (as the orchestrator: NAV_LOCATE_LOST_MARGIN)
    assert locate_vote(loc("n2", status="lost", margin=0.07), 0.02, 0.06) == "n2"
    assert locate_vote(loc("n2", status="lost", margin=0.05), 0.02, 0.06) is None
    ahead = ["n2", "n3", "n4"]  # following n1 -> n2
    assert follow_vote(loc("n2"), "n1", ahead, 0.02) == "n2"
    assert follow_vote(loc("n2", plausible=False), "n1", ahead, 0.02) == "none"  # not walked that far yet
    assert follow_vote(loc("n1"), "n1", ahead, 0.02) == "none"  # still at the source
    assert follow_vote(loc("n3"), "n1", ahead, 0.02) == "n3"  # past n2, which the camera missed
    assert follow_vote(loc("n4", plausible=False), "n1", ahead, 0.02) == "none"  # the goal seen from afar
    assert follow_vote(loc("n9", status="confirmed"), "n1", ahead, 0.02) == "elsewhere"
    assert follow_vote(loc("n9"), "n1", ahead, 0.02) == "none"  # merely leading: noise, until it lasts (lost_s)
    assert follow_vote(loc("n2", status="lost"), "n1", ahead, 0.02) == "none"  # following never counts lost


def test_locate_route_once_and_follow_hop_by_hop():
    lp = OrchestratorLoop("n4", path_from, LoopParams(k=2, n=3))
    assert lp.request() == {"previous": None, "expected": None, "previous_step_count": None}
    assert lp.on_localization(loc("n1"), 0, 1) is None  # one vote is not enough
    ev = lp.on_localization(loc("n1"), 2, 2)
    assert (ev.kind, ev.node) == ("located", "n1")
    assert lp.request() == {"previous": "n1", "expected": "n2", "previous_step_count": 2}
    # low scores are fine: the target leading, twice in the last three, is reached
    assert lp.on_localization(loc("n2", score=0.3), 5, 3) is None
    assert lp.on_localization(loc("n1"), 7, 4) is None
    ev = lp.on_localization(loc("n2", score=0.31), 9, 5)
    assert (ev.kind, ev.node) == ("reached", "n2")
    assert lp.request() == {"previous": "n2", "expected": "n3", "previous_step_count": 9}


def test_a_node_further_on_is_reached_once_walked_there_and_the_route_goes_on():
    """The camera misses n2: n3 leading is where the user is, past n2, and the route goes on from n3. Itnig,
    2026-10-10: hop by hop only, such a miss dropped the route and located the user again."""
    lp = OrchestratorLoop("n4", path_from, LoopParams(k=2, n=3))
    lp.on_localization(loc("n1", status="confirmed", margin=0.1), 0, 1)  # a confirmed result locates at once
    assert lp.on_localization(loc("n3"), 4, 2) is None
    ev = lp.on_localization(loc("n3"), 6, 3)
    assert (ev.kind, ev.node) == ("reached", "n3") and lp.path is not None
    assert lp.request() == {"previous": "n3", "expected": "n4", "previous_step_count": 6}


def test_the_goal_seen_from_afar_waits_and_a_node_off_the_route_starts_over():
    lp = OrchestratorLoop("n4", path_from, LoopParams(k=2, n=3))
    lp.on_localization(loc("n1", status="confirmed", margin=0.1), 0, 1)
    for t in (2, 3, 4):  # nav-api: the user hasn't walked that far
        assert lp.on_localization(loc("n4", plausible=False), 3, t) is None
    assert lp.path is not None and not lp.arrived
    assert lp.on_localization(loc("n9", status="confirmed"), 3, 5) is None
    ev = lp.on_localization(loc("n9", status="confirmed"), 4, 6)
    assert (ev.kind, ev.node) == ("lost", "n1") and lp.path is None and not lp.arrived
    # relocated at the goal itself: arrived
    lp.on_localization(loc("n4"), 5, 7)
    ev = lp.on_localization(loc("n4"), 6, 8)
    assert ev.kind == "arrived" and lp.arrived


def test_a_while_with_nothing_pointing_at_the_route_starts_over():
    """Lost results and nodes off the route merely leading run the clock; the hop's source or target stops it."""
    lp = OrchestratorLoop("n4", path_from, LoopParams(k=2, n=3, lost_s=3.0))
    lp.on_localization(loc("n1", status="confirmed", margin=0.1), 0, 1)
    assert lp.on_localization(loc(None, status="lost"), 0, 2) is None
    assert lp.on_localization(loc("n9"), 0, 3) is None
    assert lp.on_localization(loc("n1", margin=0.0), 0, 4.5) is None  # the source leads: the clock stops
    assert lp.on_localization(loc("n9"), 0, 5) is None
    assert lp.on_localization(loc(None, status="lost"), 0, 7.9) is None
    assert lp.on_localization(loc("n9"), 0, 8).kind == "lost"


def test_the_last_hop_arrives():
    lp = OrchestratorLoop("n2", path_from, LoopParams(k=1, n=1))
    lp.on_localization(loc("n1"), 0, 1)
    ev = lp.on_localization(loc("n2"), 4, 2)
    assert ev.kind == "arrived" and lp.arrived and lp.on_localization(loc("n2"), 5, 3) is None


def test_no_route_keeps_locating():
    lp = OrchestratorLoop("n4", lambda node, t: None, LoopParams(k=1, n=1))
    ev = lp.on_localization(loc("n1"), 0, 1)
    assert ev.kind == "no_route" and lp.path is None and lp.node == "n1"


def test_relocating_sends_the_last_node_and_ignores_nodes_too_far():
    lp = OrchestratorLoop("n4", path_from, LoopParams(k=1, n=1))
    lp.on_localization(loc("n1"), 3, 1)
    lp.path = None  # lost
    assert lp.request() == {"previous": "n1", "expected": None, "previous_step_count": 3}
    assert lp.on_localization(loc("n3", plausible=False), 4, 2) is None  # not walked that far
    assert lp.on_localization(loc("n2", plausible=True), 9, 3).node == "n2"


def test_the_walk_is_at_every_node_it_got_to_also_one_with_no_edge_from_the_last():
    """Itnig walk 601a went Corridor end (n6) -> Drinks area (n2) at 87.7 s, where the map has no edge. The replay's
    truth came from the walk's hops only, so it missed that visit; a loop arriving at the goal Drinks area at 85.4 s
    was scored as wrong (2026-10-10). walk_visits keeps it; walk_route has only the hops."""
    pmap = PlaceMap.model_validate_json((Path(__file__).parent / "fixtures" / "itnig" / "map.json").read_text())
    rec = "20261010T110802Z_601a"
    assert (87_700, "n2") in [(round(t, -2), n) for t, n in walk_visits(pmap, rec)]
    hops = [(h.source, h.target) for h, _ in walk_route(pmap, rec)]
    assert ("n9", "n6") in hops and ("n6", "n2") not in hops and ("n2", "n3") in hops
