"""Node confirmation (nav/navigation/arrival.py): votes over consecutive localizations (the confirmer's own score and
margin), the walked-distance gate (nav-api's `plausible`, else dead reckoning) decide arrival; the heading only gates
passing through; nothing is skipped; off-route is a warning; manual confirmation."""

import pytest

from nav.navigation.arrival import ArrivalParams, NodeConfirmer
from nav.navigation.pdr import GraphTracker
from nav.navigation.route import Hop
from nav.schemas.navigation import Localization, NodeMatch, RefMatch, Thresholds

TH = Thresholds(confirm_score=0.6, confirm_margin=0.05, lost_score=0.3)


def route() -> list[Hop]:
    """a -> b: 7 m east; b -> c: 7 m south; c -> d: 7 m south."""
    return [Hop("e1", "a", "b", True, 7.0, 90.0, "go east", arrival_deg=90.0),
            Hop("e2", "b", "c", True, 7.0, 180.0, "go south", arrival_deg=180.0),
            Hop("e3", "c", "d", True, 7.0, 180.0, "keep south", arrival_deg=180.0)]


def loc(best: str | None, role: str = "at", toward: str | None = None, score: float = 0.8, margin: float = 0.2,
        heading: float | None = None, plausible: bool | None = None) -> Localization:
    cands = [] if best is None else [NodeMatch(node=best, name=best.upper(), score=score, plausible=plausible, refs=[
        RefMatch(id="r1", score=score, role=role, toward=toward, image="x.jpg")])]
    return Localization(map_id="t", model="m", frames=1, status="confirmed" if score >= 0.6 else "uncertain",
                        reason="", best=best, margin=margin, candidates=cands, thresholds=TH, took_ms={},
                        heading_map_deg=heading)


RULES = ArrivalParams(k=3, n=4)  # the rule tests below count votes against 3 of 4; the default is 2 of 3 (bursts)


def confirmer(params: ArrivalParams | None = None, tracker: bool = True) -> NodeConfirmer:
    hops = route()
    return NodeConfirmer(hops, GraphTracker(hops, step_length=0.7) if tracker else None, params or RULES)


def walk(c: NodeConfirmer, steps: int, heading: float) -> list:
    return [e for e in (c.on_step(heading) for _ in range(steps)) if e]


def test_arrival_needs_votes_and_dead_reckoning():
    c = confirmer()
    assert (c.previous, c.expected) == ("a", "b")
    # b recognised from the start (seen from afar): no arrival while the gate is closed
    for t in range(4):
        assert c.on_localization(t, loc("b", "approach", heading=90)) is None
    assert c.votes.count("far") == 4
    walk(c, 9, 90.0)  # ~6.3 m of 7: now near; the votes from afar don't count
    assert c.on_localization(10, loc("b", heading=90)) is None
    assert c.on_localization(11, loc("b", heading=90)) is None
    ev = c.on_localization(12, loc("b", heading=90))  # 3 of the last 4, all cast near b
    assert ev.kind == "arrived" and ev.node == "b" and ev.hop_index == 0
    assert (c.previous, c.expected) == ("b", "c")
    assert ev.evidence["pdr"]["remaining_m"] < 2


def test_one_frame_is_not_enough_and_the_heading_does_not_block_arrival():
    c = confirmer()
    walk(c, 10, 90.0)
    assert c.on_localization(0, loc("b", heading=90)) is None
    assert c.on_localization(1, loc(None)) is None
    assert c.on_localization(2, loc("b", heading=90)) is None  # 2 of 3
    c2 = confirmer()
    walk(c2, 10, 90.0)
    evs = [c2.on_localization(t, loc("b", heading=270)) for t in range(3)]  # camera pointed back: still there
    assert evs[-1].kind == "arrived" and evs[-1].node == "b"


def test_votes_use_the_confirmers_thresholds_not_nav_apis():
    c = confirmer(ArrivalParams(k=2, n=3))
    walk(c, 10, 90.0)
    # 0.35 is below nav-api's confirm score (status "uncertain") but leads clearly: it votes
    assert c.on_localization(0, loc("b", score=0.35, margin=0.08)) is None
    assert c.on_localization(1, loc("b", score=0.35, margin=0.08)).kind == "arrived"
    c2 = confirmer(ArrivalParams(k=2, n=3))
    walk(c2, 10, 90.0)
    for t, (score, margin) in enumerate([(0.25, 0.1), (0.5, 0.02), (0.25, 0.1), (0.5, 0.02)]):
        assert c2.on_localization(t, loc("b", score=score, margin=margin)) is None  # too low, or too close


def test_navapis_walked_distance_gate_replaces_dead_reckoning():
    c = confirmer(ArrivalParams(k=2, n=3))
    walk(c, 2, 0.0)  # steps the compass says go north: the tracker has hardly progressed on an eastward hop
    assert c.on_localization(0, loc("b", "approach", plausible=False)) is None  # nav-api: not walked that far
    assert c.on_localization(1, loc("b", "approach", plausible=False)) is None
    assert list(c.votes) == ["far", "far"]
    assert c.on_localization(2, loc("b", "approach", plausible=True)) is None
    assert c.on_localization(3, loc("b", "approach", plausible=True)).kind == "arrived"


def test_a_node_next_to_the_hop_is_not_elsewhere():
    nb = {"b": {"x"}, "x": {"b"}}
    c = NodeConfirmer(route(), None, ArrivalParams(k=2, n=3), nb)
    assert all(c.on_localization(t, loc("x")) is None for t in range(4))
    assert list(c.votes) == [None, None, None]


def test_no_dead_reckoning_needs_all_n_votes():
    c = confirmer(tracker=False)
    for t in range(3):
        assert c.on_localization(t, loc("b")) is None
    assert c.on_localization(3, loc("b")).kind == "arrived"


def test_pass_through_via_outgoing_views_but_not_at_the_goal():
    c = confirmer()
    walk(c, 10, 90.0)
    events = [c.on_localization(t, loc("b", "outgoing", toward="c", heading=180)) for t in range(3)]
    assert events[-1].kind == "passed" and events[-1].node == "b"
    # outgoing views of b toward another node don't count as passing toward c
    c2 = confirmer()
    walk(c2, 10, 90.0)
    assert all(c2.on_localization(t, loc("b", "outgoing", toward="x", heading=180)) is None for t in range(4))
    # at the goal there is no next hop: you arrive, you don't pass
    c3 = NodeConfirmer(route()[-1:], None)
    assert all(c3.on_localization(t, loc("d", "outgoing", toward="x")) is None for t in range(4))


def test_nothing_is_skipped():
    c = confirmer()
    walk(c, 2, 90.0)  # 1.4 m of 7: c, the node after b, recognised clearly is not a reason to confirm anything
    for t in range(6):
        assert c.on_localization(t, loc("c", heading=180)) is None
    assert c.expected == "b" and c.votes.count("far") == 4
    walk(c, 8, 90.0)  # now at b: seeing c from there means b was reached; b is confirmed (passed), never c
    evs = [c.on_localization(10 + t, loc("c", heading=180)) for t in range(3)]
    assert evs[-1].kind == "passed" and evs[-1].node == "b" and c.expected == "c"
    assert all(e is None for e in evs[:-1])


def test_off_route_is_a_warning_once_per_hop():
    c = confirmer()
    walk(c, 5, 0.0)  # d (two nodes ahead) or any node off the route: elsewhere
    assert c.on_localization(0, loc("z")) is None
    assert c.on_localization(1, loc("z")) is None
    ev = c.on_localization(2, loc("z"))
    assert ev.kind == "off_route" and ev.node == "z"
    assert c.on_localization(3, loc("z")) is None  # said once
    walk(c, 5, 90.0)
    evs = [c.on_localization(4 + t, loc("b", heading=90)) for t in range(3)]  # confirmation went on
    assert evs[-1].kind == "arrived" and evs[-1].node == "b"
    c.reroute(route()[1:])
    assert (c.previous, c.expected) == ("b", "c")


def test_uncertain_once_when_walking_past_and_manual_confirm_relearns_step_length():
    c = confirmer()
    events = walk(c, 30, 90.0)  # 21 m on a 7 m hop, never recognised
    assert [e.kind for e in events] == ["uncertain"]
    before = c.tracker.step_length
    ev = c.confirm_manual(5.0)
    assert ev.kind == "manual" and ev.node == "b" and c.expected == "c"
    assert c.tracker.step_length < before  # 30 steps for 7 m: shorter steps than assumed


def test_whole_route_to_done():
    c = confirmer(ArrivalParams(k=2, n=3))
    for hop in route():
        walk(c, 10, hop.bearing_deg)
        evs = [c.on_localization(t, loc(hop.target, heading=hop.arrival_deg)) for t in range(2)]
        assert evs[-1] and evs[-1].kind == "arrived" and evs[-1].node == hop.target
    assert c.done and c.expected is None and c.previous == "d"
    assert c.on_localization(0, loc("d")) is None and c.confirm_manual() is None


def test_route_hops_carry_arrival_heading():
    from test_navigation import BUILD, lshape

    from nav.mapping.build import finish_map
    from nav.navigation.route import plan_route

    draft, cset, tracks = lshape()
    m = finish_map(draft, cset, BUILD, "builds/b", tracks)
    h1, h2 = plan_route(m, "n1", "n3", "observed")
    assert h1.arrival_deg == pytest.approx(90)  # e1 walked east, arrives facing east
    assert h2.forward is False and h2.arrival_deg == pytest.approx(180)  # e2 walked back n2 -> n3: south


def test_default_two_of_three_bursts():
    """The app sends a burst of 3 frames every 2 s and each burst is one vote: two of the last three confirm."""
    assert (ArrivalParams().k, ArrivalParams().n) == (2, 3)
    hops = route()
    c = NodeConfirmer(hops, GraphTracker(hops, step_length=0.7))
    walk(c, 10, 90.0)
    assert c.on_localization(0, loc("b", heading=90)) is None  # one burst is not enough
    assert c.on_localization(2, loc(None)) is None
    assert c.on_localization(4, loc("b", heading=90)).kind == "arrived"  # 2 of the last 3
    assert c.expected == "c"
