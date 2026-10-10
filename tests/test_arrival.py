"""Node confirmation (nav/navigation/arrival.py): votes over consecutive localizations, the dead-reckoning gate and the
heading decide arrival or pass-through; nothing is skipped; off-route and manual confirmation."""

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
        heading: float | None = None) -> Localization:
    cands = [] if best is None else [NodeMatch(node=best, name=best.upper(), score=score, refs=[
        RefMatch(id="r1", score=score, role=role, toward=toward, image="x.jpg")])]
    return Localization(map_id="t", model="m", frames=1, status="confirmed" if score >= 0.6 else "uncertain",
                        reason="", best=best, margin=margin, candidates=cands, thresholds=TH, took_ms={},
                        heading_map_deg=heading)


def confirmer(params: ArrivalParams | None = None, tracker: bool = True) -> NodeConfirmer:
    hops = route()
    return NodeConfirmer(hops, GraphTracker(hops, step_length=0.7) if tracker else None, params)


def walk(c: NodeConfirmer, steps: int, heading: float) -> list:
    return [e for e in (c.on_step(heading) for _ in range(steps)) if e]


def test_arrival_needs_votes_dead_reckoning_and_heading():
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


def test_one_frame_is_not_enough_and_wrong_heading_blocks_arrival():
    c = confirmer()
    walk(c, 10, 90.0)
    assert c.on_localization(0, loc("b", heading=90)) is None
    assert c.on_localization(1, loc(None)) is None
    assert c.on_localization(2, loc("b", heading=90)) is None  # 2 of 3
    c2 = confirmer()
    walk(c2, 10, 90.0)
    for t in range(4):  # recognised, near, but facing back where you came from
        assert c2.on_localization(t, loc("b", heading=270)) is None


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
    assert c.expected == "b" and c.votes.count("far") == 4 and not c.off_route
    walk(c, 8, 90.0)  # now at b: seeing c from there means b was reached; b is confirmed (passed), never c
    evs = [c.on_localization(10 + t, loc("c", heading=180)) for t in range(3)]
    assert evs[-1].kind == "passed" and evs[-1].node == "b" and c.expected == "c"
    assert all(e is None for e in evs[:-1])


def test_off_route_after_k_votes_elsewhere():
    c = confirmer()
    walk(c, 5, 0.0)  # d (two nodes ahead) or any node off the route: elsewhere
    assert c.on_localization(0, loc("z")) is None
    assert c.on_localization(1, loc("z")) is None
    ev = c.on_localization(2, loc("z"))
    assert ev.kind == "off_route" and ev.node == "z" and c.off_route
    assert c.on_localization(3, loc("b", heading=90)) is None  # stopped until reroute
    c.reroute(route()[1:])
    assert (c.previous, c.expected) == ("b", "c") and not c.off_route


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
