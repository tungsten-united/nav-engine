"""The model of the orchestrator's navigation loop (nav/navigation/replay.py, OrchestratorLoop): locate, one route,
then hop by hop on votes over /localize results; lost, start over. The route is n1 -> n2 -> n3 -> n4."""

from nav.navigation.replay import LoopParams, OrchestratorLoop, follow_vote, locate_vote
from nav.navigation.route import Hop
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
    assert follow_vote(loc("n2"), "n1", "n2", 0.02) == "at_target"
    assert follow_vote(loc("n2", plausible=False), "n1", "n2", 0.02) == "none"  # not walked that far yet
    assert follow_vote(loc("n1"), "n1", "n2", 0.02) == "none"  # still at the source
    assert follow_vote(loc("n4"), "n1", "n2", 0.02) == "elsewhere"
    assert follow_vote(loc("n4", status="lost"), "n1", "n2", 0.02) == "none"


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


def test_only_the_next_node_counts_and_lost_starts_over():
    """The goal seen from afar never skips the hops in between: it votes `elsewhere`, and lost relocates."""
    lp = OrchestratorLoop("n4", path_from, LoopParams(k=2, n=3))
    lp.on_localization(loc("n1", status="confirmed", margin=0.1), 0, 1)  # a confirmed result locates at once
    assert lp.path is not None and lp.node == "n1"
    assert lp.on_localization(loc("n4"), 3, 2) is None
    ev = lp.on_localization(loc("n4"), 4, 3)
    assert (ev.kind, ev.node) == ("lost", "n1") and lp.path is None and not lp.arrived
    # relocated at the goal itself: arrived
    lp.on_localization(loc("n4"), 5, 4)
    ev = lp.on_localization(loc("n4"), 6, 5)
    assert ev.kind == "arrived" and lp.arrived


def test_a_run_of_lost_results_starts_over():
    lp = OrchestratorLoop("n4", path_from, LoopParams(k=2, n=3, lost_calls=3))
    lp.on_localization(loc("n1", status="confirmed", margin=0.1), 0, 1)
    assert lp.on_localization(loc(None, status="lost"), 0, 2) is None
    assert lp.on_localization(loc(None, status="lost"), 0, 3) is None
    assert lp.on_localization(loc(None, status="lost"), 0, 4).kind == "lost"


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
